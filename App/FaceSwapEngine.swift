import Foundation
import UIKit
import CoreImage
import CoreGraphics
import OnnxRuntimeBindings

enum FaceSwapEngineError: LocalizedError {
    case modelMissing(String)
    case unsupportedModel(String)
    case imageCouldNotBeRead
    case faceNotFound(String)
    case landmarksUnavailable(String)
    case alignmentFailed
    case invalidEmbedding
    case invalidModelInput(String)
    case invalidModelOutput
    case imageCreationFailed

    var errorDescription: String? {
        switch self {
        case .modelMissing(let name):
            return "ملف النموذج المطلوب غير موجود: \(name)"
        case .unsupportedModel(let message):
            return message
        case .imageCouldNotBeRead:
            return "تعذر تجهيز الصورة للمعالجة."
        case .faceNotFound(let role):
            return "لم يتم العثور على وجه واضح في \(role)."
        case .landmarksUnavailable(let role):
            return "تعذر تحديد معالم الوجه في \(role). جرّب صورة أمامية بإضاءة جيدة."
        case .alignmentFailed:
            return "تعذرت محاذاة الوجه. تأكد أن الوجه واضح وغير مقصوص."
        case .invalidEmbedding:
            return "أنتج نموذج ArcFace أو محوّل الهوية متجهًا غير متوقع."
        case .invalidModelInput(let details):
            return "مداخل النموذج لا تطابق البنية المتوقعة: \(details)"
        case .invalidModelOutput:
            return "أبعاد خرج نموذج تبديل الوجه غير متوقعة."
        case .imageCreationFailed:
            return "تعذر إنشاء الصورة الناتجة."
        }
    }
}

enum FaceSwapEngine {
    private static let arcFaceModel = "arcface_w600k_r50.onnx"
    private static let embeddingLength = 512
    private static let arcFaceSize = 112

    /// Runs a local image-only swap using FaceFusion's ONNX face detector and landmarker.
    /// All model files are imported manually; Apple Vision is not used for face analysis.
    static func swap(
        source: UIImage,
        target: UIImage,
        modelDirectory: URL,
        selectedModel: FaceFusionSwapModel
    ) throws -> UIImage {
        try autoreleasepool {
            guard selectedModel.isSupportedByCurrentEngine else {
                throw FaceSwapEngineError.unsupportedModel(
                    selectedModel.supportNote ?? "هذا النموذج غير مدعوم في هذا الإصدار."
                )
            }

            for filename in selectedModel.requiredModelFiles {
                let url = modelDirectory.appendingPathComponent(filename)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    throw FaceSwapEngineError.modelMissing(filename)
                }
            }

            guard let sourceCG = normalizedCGImage(from: source),
                  let targetCG = normalizedCGImage(from: target) else {
                throw FaceSwapEngineError.imageCouldNotBeRead
            }

            let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            let options = try ORTSessionOptions()
            try options.setIntraOpNumThreads(2)

            let detectorURL = modelDirectory.appendingPathComponent(
                FaceFusionFaceAnalyzer.detectorFilename
            )
            let detectorSession = try ORTSession(
                env: env,
                modelPath: detectorURL.path,
                sessionOptions: options
            )
            let landmarkerURL = modelDirectory.appendingPathComponent(
                FaceFusionFaceAnalyzer.landmarkerFilename
            )
            let landmarkerSession = try ORTSession(
                env: env,
                modelPath: landmarkerURL.path,
                sessionOptions: options
            )

            // FaceFusion-compatible analysis: YOLO Face produces the five points
            // used for alignment; 2DFAN4 predicts the original 68-point set.
            let sourceFace = try FaceFusionFaceAnalyzer.analyze(
                image: sourceCG,
                role: "صورة المصدر",
                detectorSession: detectorSession,
                landmarkerSession: landmarkerSession
            )
            let targetFace = try FaceFusionFaceAnalyzer.analyze(
                image: targetCG,
                role: "الصورة الهدف",
                detectorSession: detectorSession,
                landmarkerSession: landmarkerSession
            )
            let sourceLandmarks = sourceFace.landmarks5CoreImage
            let targetLandmarks = targetFace.landmarks5CoreImage

            let arcFaceURL = modelDirectory.appendingPathComponent(arcFaceModel)
            let arcFaceSession = try ORTSession(
                env: env,
                modelPath: arcFaceURL.path,
                sessionOptions: options
            )

            let sourceEmbedding = try extractEmbedding(
                image: sourceCG,
                landmarks: sourceLandmarks,
                session: arcFaceSession
            )

            let sourceInput: ModelInput
            if let sourceFrameSize = selectedModel.sourceFrameSize,
               let sourceTemplate = selectedModel.sourceFrameTemplate {
                let ciContext = CIContext(options: [.cacheIntermediates: false])
                let alignedSource = try makeAlignedCrop(
                    image: CIImage(cgImage: sourceCG),
                    landmarks: sourceLandmarks,
                    template: sourceTemplate,
                    size: sourceFrameSize,
                    context: ciContext
                )
                let values = try imageToNCHW(
                    alignedSource.image,
                    size: sourceFrameSize,
                    mean: [0, 0, 0],
                    standardDeviation: [1, 1, 1]
                )
                sourceInput = .image(values, sourceFrameSize)
            } else {
                let embedding = try prepareSourceEmbedding(
                    rawEmbedding: sourceEmbedding,
                    model: selectedModel,
                    modelDirectory: modelDirectory,
                    env: env,
                    options: options
                )
                sourceInput = .embedding(embedding)
            }

            let ciContext = CIContext(options: [.cacheIntermediates: false])
            let targetCI = CIImage(cgImage: targetCG)
            let alignedTarget = try makeAlignedCrop(
                image: targetCI,
                landmarks: targetLandmarks,
                template: selectedModel.targetTemplate,
                size: selectedModel.resolution,
                context: ciContext
            )

            let targetValues = try imageToNCHW(
                alignedTarget.image,
                size: selectedModel.resolution,
                mean: selectedModel.inputMean,
                standardDeviation: selectedModel.inputStandardDeviation
            )

            let swappedValues = try runSwapModel(
                targetValues: targetValues,
                targetSize: selectedModel.resolution,
                sourceInput: sourceInput,
                model: selectedModel,
                modelDirectory: modelDirectory,
                env: env,
                options: options
            )

            guard let swappedCG = makeRGBImage(
                fromNCHW: swappedValues,
                size: selectedModel.resolution,
                denormalize: selectedModel.outputNeedsDenormalization,
                reverseChannels: selectedModel.outputNeedsChannelReversal,
                mean: selectedModel.inputMean,
                standardDeviation: selectedModel.inputStandardDeviation
            ), let maskCG = makeFeatheredMask(size: selectedModel.resolution) else {
                throw FaceSwapEngineError.imageCreationFailed
            }

            let inverseTransform = alignedTarget.transform.inverted()
            let replacementCI = CIImage(cgImage: swappedCG)
                .transformed(by: inverseTransform)
            let maskCI = CIImage(cgImage: maskCG)
                .transformed(by: inverseTransform)

            guard let blend = CIFilter(name: "CIBlendWithMask") else {
                throw FaceSwapEngineError.imageCreationFailed
            }
            blend.setValue(replacementCI, forKey: kCIInputImageKey)
            blend.setValue(targetCI, forKey: kCIInputBackgroundImageKey)
            blend.setValue(maskCI, forKey: kCIInputMaskImageKey)

            guard let blended = blend.outputImage?.cropped(to: targetCI.extent),
                  let resultCG = ciContext.createCGImage(blended, from: targetCI.extent) else {
                throw FaceSwapEngineError.imageCreationFailed
            }

            return UIImage(cgImage: resultCG, scale: target.scale, orientation: .up)
        }
    }

    private enum ModelInput {
        case embedding([Float])
        case image([Float], Int)
    }

    // MARK: - Embeddings

    private static func extractEmbedding(
        image: CGImage,
        landmarks: [CGPoint],
        session: ORTSession
    ) throws -> [Float] {
        let context = CIContext(options: [.cacheIntermediates: false])
        let aligned = try makeAlignedCrop(
            image: CIImage(cgImage: image),
            landmarks: landmarks,
            template: .arcFace112V2,
            size: arcFaceSize,
            context: context
        )
        let inputValues = try imageToNCHW(
            aligned.image,
            size: arcFaceSize,
            mean: [0.5, 0.5, 0.5],
            standardDeviation: [0.5, 0.5, 0.5]
        )

        let inputNames = try session.inputNames()
        let outputNames = try session.outputNames()
        guard let inputName = inputNames.first,
              let outputName = outputNames.first else {
            throw FaceSwapEngineError.invalidEmbedding
        }

        let tensor = try makeTensor(inputValues, shape: [1, 3, arcFaceSize, arcFaceSize])
        let outputs = try session.run(
            withInputs: [inputName: tensor],
            outputNames: [outputName],
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw FaceSwapEngineError.invalidEmbedding
        }

        let values = floats(from: try output.tensorData() as Data)
        guard values.count == embeddingLength,
              values.allSatisfy({ $0.isFinite }) else {
            throw FaceSwapEngineError.invalidEmbedding
        }
        return values
    }

    private static func prepareSourceEmbedding(
        rawEmbedding: [Float],
        model: FaceFusionSwapModel,
        modelDirectory: URL,
        env: ORTEnv,
        options: ORTSessionOptions
    ) throws -> [Float] {
        switch model.family {
        case .alphaFace:
            // Upstream AlphaFace consumes the raw ArcFace embedding.
            return rawEmbedding

        case .hyperSwap:
            // FaceFusion uses the L2-normalized ArcFace embedding for HyperSwap.
            return normalized(rawEmbedding)

        case .ghost, .hifiFace, .simSwap:
            guard let converterName = model.embeddingConverterFilename else {
                throw FaceSwapEngineError.invalidEmbedding
            }
            let converterURL = modelDirectory.appendingPathComponent(converterName)
            let converted = try runEmbeddingConverter(
                embedding: rawEmbedding,
                converterURL: converterURL,
                env: env,
                options: options
            )
            if model.family == .ghost {
                // FaceFusion keeps the converted (non-L2-normalized) Ghost embedding.
                return converted
            }
            // HifiFace and SimSwap use the normalized converted embedding.
            return normalized(converted)

        case .blendSwap, .uniFace:
            throw FaceSwapEngineError.invalidEmbedding

        case .inSwapper:
            let modelURL = modelDirectory.appendingPathComponent(model.filename)
            let initializer = try FaceFusionModelInitializer.readInSwapperEmap(from: modelURL)
            return try applyInSwapperInitializer(rawEmbedding, matrix: initializer)
        }
    }

    /// Matches FaceFusion's source_embedding = dot(embedding, emap) / norm(embedding).
    private static func applyInSwapperInitializer(
        _ embedding: [Float],
        matrix: [Float]
    ) throws -> [Float] {
        guard embedding.count == embeddingLength,
              matrix.count == embeddingLength * embeddingLength else {
            throw FaceSwapEngineError.invalidEmbedding
        }
        let norm = sqrt(embedding.reduce(Float.zero) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0 else {
            throw FaceSwapEngineError.invalidEmbedding
        }

        var output = [Float](repeating: 0, count: embeddingLength)
        for column in 0..<embeddingLength {
            var total: Float = 0
            for row in 0..<embeddingLength {
                total += embedding[row] * matrix[row * embeddingLength + column]
            }
            output[column] = total / norm
        }
        guard output.allSatisfy({ $0.isFinite }) else {
            throw FaceSwapEngineError.invalidEmbedding
        }
        return output
    }

    private static func runEmbeddingConverter(
        embedding: [Float],
        converterURL: URL,
        env: ORTEnv,
        options: ORTSessionOptions
    ) throws -> [Float] {
        let session = try ORTSession(
            env: env,
            modelPath: converterURL.path,
            sessionOptions: options
        )
        let inputNames = try session.inputNames()
        let outputNames = try session.outputNames()
        guard let inputName = inputNames.first,
              let outputName = outputNames.first else {
            throw FaceSwapEngineError.invalidEmbedding
        }
        let tensor = try makeTensor(embedding, shape: [1, embeddingLength])
        let outputs = try session.run(
            withInputs: [inputName: tensor],
            outputNames: [outputName],
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw FaceSwapEngineError.invalidEmbedding
        }
        let values = floats(from: try output.tensorData() as Data)
        guard values.count == embeddingLength,
              values.allSatisfy({ $0.isFinite }) else {
            throw FaceSwapEngineError.invalidEmbedding
        }
        return values
    }

    // MARK: - Face-swap inference

    private static func runSwapModel(
        targetValues: [Float],
        targetSize: Int,
        sourceInput: ModelInput,
        model: FaceFusionSwapModel,
        modelDirectory: URL,
        env: ORTEnv,
        options: ORTSessionOptions
    ) throws -> [Float] {
        let modelURL = modelDirectory.appendingPathComponent(model.filename)
        let session = try ORTSession(
            env: env,
            modelPath: modelURL.path,
            sessionOptions: options
        )
        let inputNames = try session.inputNames()
        let outputNames = try session.outputNames()
        guard inputNames.contains("target"),
              inputNames.contains("source"),
              let outputName = outputNames.first else {
            throw FaceSwapEngineError.invalidModelInput(
                "المداخل المكتشفة: \(inputNames.joined(separator: ", ")). المتوقع target وsource."
            )
        }

        let targetTensor = try makeTensor(
            targetValues,
            shape: [1, 3, targetSize, targetSize]
        )
        let sourceTensor: ORTValue
        let sourceShape: [Int]
        switch sourceInput {
        case .embedding(let values):
            guard values.count == embeddingLength else {
                throw FaceSwapEngineError.invalidEmbedding
            }
            sourceShape = [1, embeddingLength]
            sourceTensor = try makeTensor(values, shape: sourceShape)
        case .image(let values, let size):
            guard values.count == 3 * size * size else {
                throw FaceSwapEngineError.invalidModelInput("حجم قص صورة المصدر غير صحيح.")
            }
            sourceShape = [1, 3, size, size]
            sourceTensor = try makeTensor(values, shape: sourceShape)
        }

        let outputs = try session.run(
            withInputs: ["target": targetTensor, "source": sourceTensor],
            outputNames: [outputName],
            runOptions: nil
        )
        guard let output = outputs[outputName] else {
            throw FaceSwapEngineError.invalidModelOutput
        }

        let shape = try output.tensorTypeAndShapeInfo().shape.map { $0.intValue }
        guard shape == [1, 3, targetSize, targetSize] else {
            throw FaceSwapEngineError.invalidModelOutput
        }
        let values = floats(from: try output.tensorData() as Data)
        guard values.count == 3 * targetSize * targetSize,
              values.allSatisfy({ $0.isFinite }) else {
            throw FaceSwapEngineError.invalidModelOutput
        }
        return values
    }

    // MARK: - Face alignment

    private static func makeAlignedCrop(
        image: CIImage,
        landmarks: [CGPoint],
        template: FaceFusionFaceTemplate,
        size: Int,
        context: CIContext
    ) throws -> (image: CGImage, transform: CGAffineTransform) {
        let destinationPoints = templatePoints(template, size: size)
        let transform = try estimateSimilarityTransform(from: landmarks, to: destinationPoints)
        let cropRect = CGRect(x: 0, y: 0, width: CGFloat(size), height: CGFloat(size))
        let aligned = image.clampedToExtent()
            .transformed(by: transform)
            .cropped(to: cropRect)
        guard let cgImage = context.createCGImage(aligned, from: cropRect) else {
            throw FaceSwapEngineError.alignmentFailed
        }
        return (cgImage, transform)
    }

    private static func templatePoints(
        _ template: FaceFusionFaceTemplate,
        size: Int
    ) -> [CGPoint] {
        let normalized: [[CGFloat]]
        switch template {
        case .arcFace112V1:
            normalized = [
                [0.35473214, 0.45658929], [0.64526786, 0.45658929],
                [0.50000000, 0.61154464], [0.37913393, 0.77687500],
                [0.62086607, 0.77687500]
            ]
        case .arcFace112V2:
            normalized = [
                [0.34191607, 0.46157411], [0.65653393, 0.45983393],
                [0.50022500, 0.64050536], [0.37097589, 0.82469196],
                [0.63151696, 0.82325089]
            ]
        case .arcFace128:
            normalized = [
                [0.36167656, 0.40387734], [0.63696719, 0.40235469],
                [0.50019687, 0.56044219], [0.38710391, 0.72160547],
                [0.61507734, 0.72034453]
            ]
        case .ffhq512:
            normalized = [
                [0.37691676, 0.46864664], [0.62285697, 0.46912813],
                [0.50123859, 0.61331904], [0.39308822, 0.72541100],
                [0.61150205, 0.72490465]
            ]
        case .mtcnn512:
            normalized = [
                [0.36562865, 0.46733799], [0.63305391, 0.46585885],
                [0.50019127, 0.61942959], [0.39032951, 0.77598822],
                [0.61178945, 0.77476328]
            ]
        }
        let dimension = CGFloat(size)
        // The model templates use top-left image coordinates; Core Image uses bottom-left.
        return normalized.map { CGPoint(x: $0[0] * dimension, y: (1 - $0[1]) * dimension) }
    }

    private static func estimateSimilarityTransform(
        from source: [CGPoint],
        to destination: [CGPoint]
    ) throws -> CGAffineTransform {
        guard source.count == destination.count, source.count >= 3 else {
            throw FaceSwapEngineError.alignmentFailed
        }
        let count = CGFloat(source.count)
        let sourceMean = CGPoint(
            x: source.reduce(CGFloat.zero) { $0 + $1.x } / count,
            y: source.reduce(CGFloat.zero) { $0 + $1.y } / count
        )
        let destinationMean = CGPoint(
            x: destination.reduce(CGFloat.zero) { $0 + $1.x } / count,
            y: destination.reduce(CGFloat.zero) { $0 + $1.y } / count
        )
        var denominator: CGFloat = 0
        var numeratorA: CGFloat = 0
        var numeratorB: CGFloat = 0
        for index in 0..<source.count {
            let x = source[index].x - sourceMean.x
            let y = source[index].y - sourceMean.y
            let u = destination[index].x - destinationMean.x
            let v = destination[index].y - destinationMean.y
            denominator += x * x + y * y
            numeratorA += x * u + y * v
            numeratorB += x * v - y * u
        }
        guard denominator > 0 else { throw FaceSwapEngineError.alignmentFailed }
        let a = numeratorA / denominator
        let b = numeratorB / denominator
        let tx = destinationMean.x - (a * sourceMean.x - b * sourceMean.y)
        let ty = destinationMean.y - (b * sourceMean.x + a * sourceMean.y)
        guard a.isFinite, b.isFinite, tx.isFinite, ty.isFinite,
              a * a + b * b > 0.0000001 else {
            throw FaceSwapEngineError.alignmentFailed
        }
        return CGAffineTransform(a: a, b: b, c: -b, d: a, tx: tx, ty: ty)
    }

    // MARK: - Tensor and image conversion

    private static func imageToNCHW(
        _ image: CGImage,
        size: Int,
        mean: [Float],
        standardDeviation: [Float]
    ) throws -> [Float] {
        guard image.width == size, image.height == size,
              mean.count == 3, standardDeviation.count == 3,
              standardDeviation.allSatisfy({ $0 != 0 }) else {
            throw FaceSwapEngineError.invalidModelInput("حجم الصورة أو إعدادات التطبيع غير صحيحة.")
        }
        let pixels = rgbaPixels(from: image)
        let planeSize = size * size
        var tensor = [Float](repeating: 0, count: 3 * planeSize)
        for index in 0..<planeSize {
            let offset = index * 4
            let alpha = Float(pixels[offset + 3]) / 255
            let red = alpha > 0 ? min(1, (Float(pixels[offset]) / 255) / alpha) : 0
            let green = alpha > 0 ? min(1, (Float(pixels[offset + 1]) / 255) / alpha) : 0
            let blue = alpha > 0 ? min(1, (Float(pixels[offset + 2]) / 255) / alpha) : 0
            tensor[index] = (red - mean[0]) / standardDeviation[0]
            tensor[planeSize + index] = (green - mean[1]) / standardDeviation[1]
            tensor[2 * planeSize + index] = (blue - mean[2]) / standardDeviation[2]
        }
        return tensor
    }

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
        data.withUnsafeBytes { rawBuffer in Array(rawBuffer.bindMemory(to: Float.self)) }
    }

    private static func normalized(_ values: [Float]) -> [Float] {
        let norm = sqrt(values.reduce(Float.zero) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0 else { return values }
        return values.map { $0 / norm }
    }

    private static func normalizedCGImage(from image: UIImage) -> CGImage? {
        if image.imageOrientation == .up, let cgImage = image.cgImage { return cgImage }
        let pixelSize = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        guard pixelSize.width > 0, pixelSize.height > 0 else { return nil }
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: pixelSize, format: format)
        return renderer.image { _ in image.draw(in: CGRect(origin: .zero, size: pixelSize)) }.cgImage
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

    private static func makeRGBImage(
        fromNCHW values: [Float],
        size: Int,
        denormalize: Bool,
        reverseChannels: Bool,
        mean: [Float],
        standardDeviation: [Float]
    ) -> CGImage? {
        let planeSize = size * size
        guard values.count == 3 * planeSize else { return nil }
        var rgba = [UInt8](repeating: 255, count: planeSize * 4)
        for index in 0..<planeSize {
            func displayValue(_ value: Float, channel: Int) -> Float {
                let denormalized = denormalize
                    ? value * standardDeviation[channel] + mean[channel]
                    : value
                return min(1, max(0, denormalized))
            }
            let modelRed = displayValue(values[index], channel: 0)
            let green = displayValue(values[planeSize + index], channel: 1)
            let modelBlue = displayValue(values[2 * planeSize + index], channel: 2)
            let red = reverseChannels ? modelBlue : modelRed
            let blue = reverseChannels ? modelRed : modelBlue
            let offset = index * 4
            rgba[offset] = UInt8((red * 255).rounded())
            rgba[offset + 1] = UInt8((green * 255).rounded())
            rgba[offset + 2] = UInt8((blue * 255).rounded())
            rgba[offset + 3] = 255
        }
        let data = Data(rgba) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        let bitmapInfo = CGBitmapInfo(
            rawValue: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
        )
        return CGImage(
            width: size,
            height: size,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    private static func makeFeatheredMask(size: Int) -> CGImage? {
        var bytes = [UInt8](repeating: 0, count: size * size)
        let centerX = Double(size) / 2
        let centerY = Double(size) / 2
        let radiusX = Double(size) * 0.46
        let radiusY = Double(size) * 0.49
        let solidUntil = 0.72
        for y in 0..<size {
            for x in 0..<size {
                let dx = (Double(x) + 0.5 - centerX) / radiusX
                let dy = (Double(y) + 0.5 - centerY) / radiusY
                let radius = sqrt(dx * dx + dy * dy)
                let opacity: Double
                if radius <= solidUntil {
                    opacity = 1
                } else if radius >= 1 {
                    opacity = 0
                } else {
                    let value = (1 - radius) / (1 - solidUntil)
                    opacity = value * value * (3 - 2 * value)
                }
                bytes[y * size + x] = UInt8((opacity * 255).rounded())
            }
        }
        let data = Data(bytes) as CFData
        guard let provider = CGDataProvider(data: data) else { return nil }
        return CGImage(
            width: size,
            height: size,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: size,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}
