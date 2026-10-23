import Foundation
import UIKit
import CoreGraphics
import OnnxRuntimeBindings

/// Local iOS port of FaceFusion's default face-analysis path:
/// YOLO Face (5-point detector) followed by 2DFAN4 (68-point landmarker).
/// The model assets must be imported by the user; this type never downloads them.
enum FaceFusionFaceAnalyzer {
    static let detectorFilename = "yoloface_8n.onnx"
    static let landmarkerFilename = "2dfan4.onnx"
    static let detectorSize = 640
    static let landmarkerSize = 256
    private static let faceScoreThreshold: Float = 0.5

    static var requiredModelFiles: [String] {
        [detectorFilename, landmarkerFilename]
    }

    struct DetectedFace {
        /// Pixel-coordinate box with origin at the top-left of the source image.
        let boundingBox: CGRect
        /// Detector's five points in Core Image coordinates (origin bottom-left).
        let landmarks5CoreImage: [CGPoint]
        /// 68-point landmarks in source-image top-left pixel coordinates.
        let landmarks68TopLeft: [CGPoint]
        let confidence: Float
    }

    private struct DetectionCandidate {
        let box: CGRect
        let score: Float
        let landmarksTopLeft: [CGPoint]
    }

    static func analyze(
        image: CGImage,
        role: String,
        detectorSession: ORTSession,
        landmarkerSession: ORTSession
    ) throws -> DetectedFace {
        let candidate = try detectLargestFace(
            image: image,
            role: role,
            session: detectorSession
        )
        let landmarks68 = try detect68Landmarks(
            image: image,
            box: candidate.box,
            role: role,
            session: landmarkerSession
        )

        let coreImagePoints = candidate.landmarksTopLeft.map { point in
            CGPoint(x: point.x, y: CGFloat(image.height) - point.y)
        }
        return DetectedFace(
            boundingBox: candidate.box,
            landmarks5CoreImage: coreImagePoints,
            landmarks68TopLeft: landmarks68,
            confidence: candidate.score
        )
    }

    // MARK: FaceFusion face_detector.py — detect_with_yolo_face

    private static func detectLargestFace(
        image: CGImage,
        role: String,
        session: ORTSession
    ) throws -> DetectionCandidate {
        let imageWidth = image.width
        let imageHeight = image.height
        let resized = try resizeKeepingAspect(image, maxDimension: detectorSize)
        let resizedWidth = resized.width
        let resizedHeight = resized.height
        let pixels = rgbaPixels(from: resized)
        let plane = detectorSize * detectorSize
        var input = [Float](repeating: 0, count: 3 * plane)

        // FaceFusion's OpenCV frames are BGR; prepare_detect_frame places the
        // restricted frame at the top-left of a zero-filled 640x640 tensor.
        for y in 0..<resizedHeight {
            for x in 0..<resizedWidth {
                let pixelOffset = (y * resizedWidth + x) * 4
                let tensorOffset = y * detectorSize + x
                let red = Float(pixels[pixelOffset]) / 255.0
                let green = Float(pixels[pixelOffset + 1]) / 255.0
                let blue = Float(pixels[pixelOffset + 2]) / 255.0
                input[tensorOffset] = blue
                input[plane + tensorOffset] = green
                input[2 * plane + tensorOffset] = red
            }
        }

        guard let inputName = try session.inputNames().first,
              let outputName = try session.outputNames().first else {
            throw FaceSwapEngineError.invalidModelInput("تعذر قراءة أسماء مداخل/مخارج yoloface_8n.onnx.")
        }
        let tensor = try makeTensor(input, shape: [1, 3, detectorSize, detectorSize])
        let outputs = try session.run(
            withInputs: [inputName: tensor],
            outputNames: [outputName],
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw FaceSwapEngineError.invalidModelOutput
        }

        let shape = try output.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        let values = floats(from: try output.tensorData() as Data)
        guard values.count >= 20 * 100,
              values.count % 20 == 0,
              values.allSatisfy({ $0.isFinite }) else {
            throw FaceSwapEngineError.invalidModelOutput
        }

        // Official FaceFusion YOLO Face output is normally [1, 20, N]. Accept
        // [1, N, 20] too so compatible exports can be inspected and tested.
        let trailing = Array(shape.suffix(2))
        let channelFirst: Bool
        if trailing.count == 2, trailing[0] == 20, trailing[1] > 20 {
            channelFirst = true
        } else if trailing.count == 2, trailing[1] == 20, trailing[0] > 20 {
            channelFirst = false
        } else {
            // Shape metadata can be dynamic (-1) in some exports. FaceFusion's
            // detector has 20 values per candidate, so infer the common layout.
            channelFirst = trailing.count == 2 && trailing[0] > trailing[1]
        }
        let candidateCount = values.count / 20
        let scaleX = CGFloat(imageWidth) / CGFloat(resizedWidth)
        let scaleY = CGFloat(imageHeight) / CGFloat(resizedHeight)

        func value(candidate: Int, channel: Int) -> Float {
            channelFirst
                ? values[channel * candidateCount + candidate]
                : values[candidate * 20 + channel]
        }

        var candidates: [DetectionCandidate] = []
        candidates.reserveCapacity(8)

        // Mirrors FaceFusion's official decoder: xywh, one confidence value,
        // then five (x, y, confidence) landmark triplets. It thresholds scores
        // without adding a different detector implementation or Apple's Vision.
        for index in 0..<candidateCount {
            let score = value(candidate: index, channel: 4)
            guard score > faceScoreThreshold else { continue }

            let centerX = CGFloat(value(candidate: index, channel: 0))
            let centerY = CGFloat(value(candidate: index, channel: 1))
            let width = CGFloat(value(candidate: index, channel: 2))
            let height = CGFloat(value(candidate: index, channel: 3))
            guard centerX.isFinite, centerY.isFinite,
                  width.isFinite, height.isFinite, width > 0, height > 0,
                  centerX >= 0, centerY >= 0,
                  centerX < CGFloat(resizedWidth),
                  centerY < CGFloat(resizedHeight) else { continue }

            let minX = max(0, min(CGFloat(imageWidth), (centerX - width / 2) * scaleX))
            let maxX = max(0, min(CGFloat(imageWidth), (centerX + width / 2) * scaleX))
            let minY = max(0, min(CGFloat(imageHeight), (centerY - height / 2) * scaleY))
            let maxY = max(0, min(CGFloat(imageHeight), (centerY + height / 2) * scaleY))
            guard maxX > minX, maxY > minY else { continue }

            var landmarks: [CGPoint] = []
            landmarks.reserveCapacity(5)
            for pointIndex in 0..<5 {
                let x = CGFloat(value(candidate: index, channel: 5 + pointIndex * 3)) * scaleX
                let y = CGFloat(value(candidate: index, channel: 6 + pointIndex * 3)) * scaleY
                guard x.isFinite, y.isFinite else { continue }
                landmarks.append(
                    CGPoint(
                        x: max(0, min(CGFloat(imageWidth), x)),
                        y: max(0, min(CGFloat(imageHeight), y))
                    )
                )
            }
            guard landmarks.count == 5 else { continue }
            candidates.append(
                DetectionCandidate(
                    box: CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY),
                    score: score,
                    landmarksTopLeft: landmarks
                )
            )
        }

        guard let largest = candidates.max(by: {
            ($0.box.width * $0.box.height) < ($1.box.width * $1.box.height)
        }) else {
            throw FaceSwapEngineError.faceNotFound(role)
        }
        return largest
    }

    // MARK: FaceFusion face_landmarker.py — detect_with_2dfan4

    private static func detect68Landmarks(
        image: CGImage,
        box: CGRect,
        role: String,
        session: ORTSession
    ) throws -> [CGPoint] {
        let maxSide = max(box.width, box.height)
        guard maxSide > 1 else { throw FaceSwapEngineError.landmarksUnavailable(role) }
        let scale = 195.0 / maxSide
        let translationX = (CGFloat(landmarkerSize) - (box.minX + box.maxX) * scale) * 0.5
        let translationY = (CGFloat(landmarkerSize) - (box.minY + box.maxY) * scale) * 0.5

        // Equivalent to FaceFusion's warp_face_by_translation for face_angle=0.
        let rendererFormat = UIGraphicsImageRendererFormat()
        rendererFormat.scale = 1
        rendererFormat.opaque = true
        let renderer = UIGraphicsImageRenderer(
            size: CGSize(width: landmarkerSize, height: landmarkerSize),
            format: rendererFormat
        )
        let sourceImage = UIImage(cgImage: image, scale: 1, orientation: .up)
        let crop = renderer.image { rendererContext in
            rendererContext.cgContext.interpolationQuality = .high
            rendererContext.cgContext.setFillColor(UIColor.black.cgColor)
            rendererContext.cgContext.fill(
                CGRect(x: 0, y: 0, width: landmarkerSize, height: landmarkerSize)
            )
            sourceImage.draw(
                in: CGRect(
                    x: translationX,
                    y: translationY,
                    width: CGFloat(image.width) * scale,
                    height: CGFloat(image.height) * scale
                )
            )
        }
        guard let cropCG = crop.cgImage else {
            throw FaceSwapEngineError.landmarksUnavailable(role)
        }
        let cropPixels = rgbaPixels(from: cropCG)
        let plane = landmarkerSize * landmarkerSize
        var input = [Float](repeating: 0, count: 3 * plane)
        for index in 0..<plane {
            let offset = index * 4
            let red = Float(cropPixels[offset]) / 255.0
            let green = Float(cropPixels[offset + 1]) / 255.0
            let blue = Float(cropPixels[offset + 2]) / 255.0
            // The original FaceFusion analysis frames are BGR, normalized to [0, 1].
            input[index] = blue
            input[plane + index] = green
            input[2 * plane + index] = red
        }

        guard let inputName = try session.inputNames().first else {
            throw FaceSwapEngineError.invalidModelInput("تعذر قراءة مدخل 2dfan4.onnx.")
        }
        let outputNames = try session.outputNames()
        guard let landmarkOutputName = outputNames.first(where: {
            $0.lowercased().contains("landmark")
        }) ?? outputNames.first else {
            throw FaceSwapEngineError.invalidModelOutput
        }
        let tensor = try makeTensor(input, shape: [1, 3, landmarkerSize, landmarkerSize])
        let outputs = try session.run(
            withInputs: [inputName: tensor],
            outputNames: [landmarkOutputName],
            runOptions: nil
        )
        guard let output = outputs[landmarkOutputName] else {
            throw FaceSwapEngineError.invalidModelOutput
        }
        let values = floats(from: try output.tensorData() as Data)
        guard values.count >= 68 * 2, values.count % 68 == 0,
              values.allSatisfy({ $0.isFinite }) else {
            throw FaceSwapEngineError.invalidModelOutput
        }
        let stride = values.count / 68
        guard stride >= 2 else { throw FaceSwapEngineError.invalidModelOutput }

        var landmarks: [CGPoint] = []
        landmarks.reserveCapacity(68)
        for index in 0..<68 {
            // FaceFusion's 2DFAN4 wrapper scales its x/y output from a 64x64 map
            // back to 256x256, then applies the inverse affine transform.
            let xInCrop = CGFloat(values[index * stride]) / 64.0 * CGFloat(landmarkerSize)
            let yInCrop = CGFloat(values[index * stride + 1]) / 64.0 * CGFloat(landmarkerSize)
            let x = (xInCrop - translationX) / scale
            let y = (yInCrop - translationY) / scale
            guard x.isFinite, y.isFinite else {
                throw FaceSwapEngineError.landmarksUnavailable(role)
            }
            landmarks.append(
                CGPoint(
                    x: max(0, min(CGFloat(image.width), x)),
                    y: max(0, min(CGFloat(image.height), y))
                )
            )
        }
        return landmarks
    }

    // MARK: Tensor / image helpers

    private static func makeTensor(_ values: [Float], shape: [Int]) throws -> ORTValue {
        let data = values.withUnsafeBufferPointer { buffer -> Data in
            guard let baseAddress = buffer.baseAddress else { return Data() }
            return Data(bytes: baseAddress, count: buffer.count * MemoryLayout<Float>.size)
        }
        return try ORTValue(
            tensorData: NSMutableData(data: data),
            elementType: ORTTensorElementDataType.float,
            shape: shape.map { NSNumber(value: $0) }
        )
    }

    private static func floats(from data: Data) -> [Float] {
        data.withUnsafeBytes { rawBuffer in
            Array(rawBuffer.bindMemory(to: Float.self))
        }
    }

    private static func resizeKeepingAspect(_ image: CGImage, maxDimension: Int) throws -> CGImage {
        let factor = min(
            1.0,
            min(
                CGFloat(maxDimension) / CGFloat(image.width),
                CGFloat(maxDimension) / CGFloat(image.height)
            )
        )
        let width = max(1, Int((CGFloat(image.width) * factor).rounded()))
        let height = max(1, Int((CGFloat(image.height) * factor).rounded()))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: width, height: height), format: format)
        let resized = renderer.image { context in
            context.cgContext.interpolationQuality = .high
            UIImage(cgImage: image, scale: 1, orientation: .up)
                .draw(in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        guard let result = resized.cgImage else {
            throw FaceSwapEngineError.imageCouldNotBeRead
        }
        return result
    }

    private static func rgbaPixels(from image: CGImage) -> [UInt8] {
        let width = image.width
        let height = image.height
        let bytesPerRow = width * 4
        var bytes = [UInt8](repeating: 0, count: height * bytesPerRow)
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        )
        bytes.withUnsafeMutableBytes { buffer in
            guard let address = buffer.baseAddress,
                  let context = CGContext(
                    data: address,
                    width: width,
                    height: height,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: bitmapInfo.rawValue
                  ) else { return }
            context.translateBy(x: 0, y: CGFloat(height))
            context.scaleBy(x: 1, y: -1)
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return bytes
    }
}
