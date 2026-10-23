import Foundation

/// Small, read-only ONNX protobuf reader for the `emap` initializer used by
/// upstream InSwapper. It memory-maps the large model file and materializes only
/// the 512x512 identity mapping matrix, rather than loading all model weights.
enum FaceFusionModelInitializer {
    enum ReadError: LocalizedError {
        case initializerNotFound
        case unsupportedTensorType(Int)
        case malformedONNX
        case invalidInitializerShape

        var errorDescription: String? {
            switch self {
            case .initializerNotFound:
                return "لم يتم العثور على مصفوفة emap داخل ملف InSwapper. تأكد أن الملف من أوزان FaceFusion/InsightFace المتوافقة."
            case .unsupportedTensorType(let type):
                return "نوع بيانات مصفوفة emap غير مدعوم: ONNX TensorProto data_type=\(type)."
            case .malformedONNX:
                return "ملف ONNX تالف أو لا يمكن قراءة بنية protobuf الخاصة به."
            case .invalidInitializerShape:
                return "مصفوفة emap لا تطابق الأبعاد المتوقعة 512×512."
            }
        }
    }

    private struct Field {
        let number: Int
        let wireType: Int
        let payloadOffset: Int
        let payloadLength: Int
        let scalar: UInt64?
        let nextOffset: Int
    }

    private struct TensorInitializer {
        var name = ""
        var dimensions: [Int] = []
        var dataType: Int = 0
        var rawDataRange: Range<Int>?
        var floatValues: [Float] = []
    }

    static func readInSwapperEmap(from modelURL: URL) throws -> [Float] {
        let mappedData = try Data(contentsOf: modelURL, options: [.mappedIfSafe])
        return try mappedData.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            let fileEnd = bytes.count
            var offset = 0

            while let field = try nextField(bytes, offset: &offset, limit: fileEnd) {
                if field.number == 7 && field.wireType == 2 {
                    let graphStart = field.payloadOffset
                    let graphEnd = graphStart + field.payloadLength
                    if let matrix = try findEmapInGraph(bytes, start: graphStart, end: graphEnd) {
                        return matrix
                    }
                }
                offset = field.nextOffset
            }
            throw ReadError.initializerNotFound
        }
    }

    private static func findEmapInGraph(
        _ bytes: UnsafeBufferPointer<UInt8>,
        start: Int,
        end: Int
    ) throws -> [Float]? {
        var offset = start
        while let field = try nextField(bytes, offset: &offset, limit: end) {
            // GraphProto.initializer is field number 5.
            if field.number == 5 && field.wireType == 2 {
                let tensor = try parseTensor(
                    bytes,
                    start: field.payloadOffset,
                    end: field.payloadOffset + field.payloadLength
                )
                if tensor.name == "emap" {
                    let values = try decodeTensor(tensor, bytes: bytes)
                    guard tensor.dimensions == [512, 512], values.count == 512 * 512 else {
                        throw ReadError.invalidInitializerShape
                    }
                    return values
                }
            }
            offset = field.nextOffset
        }
        return nil
    }

    private static func parseTensor(
        _ bytes: UnsafeBufferPointer<UInt8>,
        start: Int,
        end: Int
    ) throws -> TensorInitializer {
        var tensor = TensorInitializer()
        var offset = start

        while let field = try nextField(bytes, offset: &offset, limit: end) {
            switch (field.number, field.wireType) {
            case (1, 0): // TensorProto.dims, unpacked int64
                if let scalar = field.scalar { tensor.dimensions.append(Int(scalar)) }
            case (1, 2): // TensorProto.dims, packed int64
                var packedOffset = field.payloadOffset
                let packedEnd = packedOffset + field.payloadLength
                while packedOffset < packedEnd {
                    tensor.dimensions.append(Int(try readVarint(bytes, offset: &packedOffset, limit: packedEnd)))
                }
            case (2, 0): // TensorProto.data_type
                tensor.dataType = Int(field.scalar ?? 0)
            case (4, 2): // TensorProto.float_data packed fixed32 values
                guard field.payloadLength % 4 == 0 else { throw ReadError.malformedONNX }
                var valueOffset = field.payloadOffset
                while valueOffset < field.payloadOffset + field.payloadLength {
                    tensor.floatValues.append(readFloat32(bytes, at: valueOffset))
                    valueOffset += 4
                }
            case (4, 5): // TensorProto.float_data unpacked fixed32
                tensor.floatValues.append(readFloat32(bytes, at: field.payloadOffset))
            case (8, 2): // TensorProto.name
                let range = field.payloadOffset..<(field.payloadOffset + field.payloadLength)
                tensor.name = String(decoding: bytes[range], as: UTF8.self)
            case (9, 2): // TensorProto.raw_data; keep only its byte range
                tensor.rawDataRange = field.payloadOffset..<(field.payloadOffset + field.payloadLength)
            default:
                break
            }
            offset = field.nextOffset
        }
        return tensor
    }

    private static func decodeTensor(
        _ tensor: TensorInitializer,
        bytes: UnsafeBufferPointer<UInt8>
    ) throws -> [Float] {
        if let range = tensor.rawDataRange {
            switch tensor.dataType {
            case 1: // TensorProto.FLOAT
                guard range.count % 4 == 0 else { throw ReadError.malformedONNX }
                var values = [Float]()
                values.reserveCapacity(range.count / 4)
                var offset = range.lowerBound
                while offset < range.upperBound {
                    values.append(readFloat32(bytes, at: offset))
                    offset += 4
                }
                return values
            case 10: // TensorProto.FLOAT16
                guard range.count % 2 == 0 else { throw ReadError.malformedONNX }
                var values = [Float]()
                values.reserveCapacity(range.count / 2)
                var offset = range.lowerBound
                while offset < range.upperBound {
                    let bits = UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
                    values.append(Float(Float16(bitPattern: bits)))
                    offset += 2
                }
                return values
            default:
                throw ReadError.unsupportedTensorType(tensor.dataType)
            }
        }

        guard tensor.dataType == 1 else {
            throw ReadError.unsupportedTensorType(tensor.dataType)
        }
        return tensor.floatValues
    }

    private static func nextField(
        _ bytes: UnsafeBufferPointer<UInt8>,
        offset: inout Int,
        limit: Int
    ) throws -> Field? {
        if offset >= limit { return nil }
        var cursor = offset
        let tag = try readVarint(bytes, offset: &cursor, limit: limit)
        let number = Int(tag >> 3)
        let wireType = Int(tag & 0x07)
        guard number > 0 else { throw ReadError.malformedONNX }

        let payloadOffset: Int
        let payloadLength: Int
        let scalar: UInt64?
        switch wireType {
        case 0:
            scalar = try readVarint(bytes, offset: &cursor, limit: limit)
            payloadOffset = cursor
            payloadLength = 0
        case 1:
            guard cursor <= limit - 8 else { throw ReadError.malformedONNX }
            scalar = nil
            payloadOffset = cursor
            payloadLength = 8
            cursor += 8
        case 2:
            let length = try readVarint(bytes, offset: &cursor, limit: limit)
            guard length <= UInt64(limit - cursor) else { throw ReadError.malformedONNX }
            scalar = nil
            payloadOffset = cursor
            payloadLength = Int(length)
            cursor += Int(length)
        case 5:
            guard cursor <= limit - 4 else { throw ReadError.malformedONNX }
            scalar = nil
            payloadOffset = cursor
            payloadLength = 4
            cursor += 4
        default:
            // ONNX model graph fields do not require protobuf groups (wire types 3/4).
            throw ReadError.malformedONNX
        }
        offset = cursor
        return Field(
            number: number,
            wireType: wireType,
            payloadOffset: payloadOffset,
            payloadLength: payloadLength,
            scalar: scalar,
            nextOffset: cursor
        )
    }

    private static func readVarint(
        _ bytes: UnsafeBufferPointer<UInt8>,
        offset: inout Int,
        limit: Int
    ) throws -> UInt64 {
        var result: UInt64 = 0
        var shift: UInt64 = 0
        while offset < limit && shift < 64 {
            let byte = UInt64(bytes[offset])
            offset += 1
            result |= (byte & 0x7f) << shift
            if (byte & 0x80) == 0 { return result }
            shift += 7
        }
        throw ReadError.malformedONNX
    }

    private static func readFloat32(_ bytes: UnsafeBufferPointer<UInt8>, at offset: Int) -> Float {
        let bits = UInt32(bytes[offset]) |
            (UInt32(bytes[offset + 1]) << 8) |
            (UInt32(bytes[offset + 2]) << 16) |
            (UInt32(bytes[offset + 3]) << 24)
        return Float(bitPattern: bits)
    }
}
