import Foundation

/// Face-swap model catalogue aligned with the model names/configuration used by
/// the upstream FaceFusion face_swapper module. No model is downloaded by this app.
enum FaceFusionSwapModel: String, CaseIterable, Identifiable, Hashable {
    case alphaface256 = "alphaface_256"
    case blendswap256 = "blendswap_256"
    case ghost1_256 = "ghost_1_256"
    case ghost2_256 = "ghost_2_256"
    case ghost3_256 = "ghost_3_256"
    case hififace256 = "hififace_unofficial_256"
    case hyperswap1a256 = "hyperswap_1a_256"
    case hyperswap1b256 = "hyperswap_1b_256"
    case hyperswap1c256 = "hyperswap_1c_256"
    case inswapper128 = "inswapper_128"
    case inswapper128FP16 = "inswapper_128_fp16"
    case simswap256 = "simswap_256"
    case simswap512 = "simswap_unofficial_512"
    case uniface256 = "uniface_256"

    var id: String { rawValue }
    var filename: String { rawValue + ".onnx" }

    var displayName: String {
        switch self {
        case .alphaface256: return "AlphaFace 256"
        case .blendswap256: return "BlendSwap 256"
        case .ghost1_256: return "Ghost 1 · 256"
        case .ghost2_256: return "Ghost 2 · 256"
        case .ghost3_256: return "Ghost 3 · 256"
        case .hififace256: return "HifiFace · 256"
        case .hyperswap1a256: return "HyperSwap 1A · 256"
        case .hyperswap1b256: return "HyperSwap 1B · 256"
        case .hyperswap1c256: return "HyperSwap 1C · 256"
        case .inswapper128: return "InSwapper 128"
        case .inswapper128FP16: return "InSwapper 128 FP16"
        case .simswap256: return "SimSwap 256"
        case .simswap512: return "SimSwap unofficial 512"
        case .uniface256: return "UniFace 256"
        }
    }

    var family: FaceFusionModelFamily {
        switch self {
        case .alphaface256: return .alphaFace
        case .blendswap256: return .blendSwap
        case .ghost1_256, .ghost2_256, .ghost3_256: return .ghost
        case .hififace256: return .hifiFace
        case .hyperswap1a256, .hyperswap1b256, .hyperswap1c256: return .hyperSwap
        case .inswapper128, .inswapper128FP16: return .inSwapper
        case .simswap256, .simswap512: return .simSwap
        case .uniface256: return .uniFace
        }
    }

    var resolution: Int {
        switch self {
        case .inswapper128, .inswapper128FP16: return 128
        case .simswap512: return 512
        default: return 256
        }
    }

    var targetTemplate: FaceFusionFaceTemplate {
        switch self {
        case .alphaface256, .hyperswap1a256, .hyperswap1b256,
             .hyperswap1c256, .inswapper128, .inswapper128FP16:
            return .arcFace128
        case .blendswap256, .uniface256:
            return .ffhq512
        case .ghost1_256, .ghost2_256, .ghost3_256,
             .simswap256, .simswap512:
            return .arcFace112V1
        case .hififace256:
            return .mtcnn512
        }
    }

    /// Channel-wise preprocessing used by FaceFusion for the target crop.
    var inputMean: [Float] {
        switch family {
        case .hyperSwap, .ghost, .hifiFace, .uniFace:
            return [0.5, 0.5, 0.5]
        case .simSwap where self == .simswap256:
            return [0.485, 0.456, 0.406]
        default:
            return [0, 0, 0]
        }
    }

    var inputStandardDeviation: [Float] {
        switch family {
        case .hyperSwap, .ghost, .hifiFace, .uniFace:
            return [0.5, 0.5, 0.5]
        case .simSwap where self == .simswap256:
            return [0.229, 0.224, 0.225]
        default:
            return [1, 1, 1]
        }
    }

    /// The upstream implementation converts these model outputs back from the
    /// standardized input range before compositing them into the original frame.
    var outputNeedsDenormalization: Bool {
        switch family {
        case .ghost, .hifiFace, .hyperSwap, .uniFace:
            return true
        default:
            return false
        }
    }

    /// The upstream postprocessor reverses RGB outputs to BGR for its OpenCV
    /// frames. This port composites into RGB/CoreGraphics images, so no reversal
    /// is required here after the optional denormalization.
    var outputNeedsChannelReversal: Bool { false }

    /// Models that take a source image crop rather than an identity embedding.
    var sourceFrameSize: Int? {
        switch family {
        case .blendSwap: return 112
        case .uniFace: return 256
        default: return nil
        }
    }

    var sourceFrameTemplate: FaceFusionFaceTemplate? {
        switch family {
        case .blendSwap: return .arcFace112V2
        case .uniFace: return .ffhq512
        default: return nil
        }
    }

    /// Original FaceFusion uses crossface embedding converters for these families.
    var embeddingConverterFilename: String? {
        switch family {
        case .ghost: return "crossface_ghost.onnx"
        case .hifiFace: return "crossface_hififace.onnx"
        case .simSwap: return "crossface_simswap.onnx"
        default: return nil
        }
    }

    var requiredModelFiles: [String] {
        // These are the default FaceFusion face-analysis assets, not Apple's Vision.
        var files = FaceFusionFaceAnalyzer.requiredModelFiles
        files.append(contentsOf: ["arcface_w600k_r50.onnx", filename])
        if let converter = embeddingConverterFilename {
            files.append(converter)
        }
        return files
    }

    /// InSwapper's `emap` matrix is read from the manually imported ONNX file.
    var isSupportedByCurrentEngine: Bool { true }

    var supportNote: String? { nil }
}

enum FaceFusionModelFamily: Equatable {
    case alphaFace
    case blendSwap
    case ghost
    case hifiFace
    case hyperSwap
    case inSwapper
    case simSwap
    case uniFace
}

enum FaceFusionFaceTemplate {
    case arcFace112V1
    case arcFace112V2
    case arcFace128
    case ffhq512
    case mtcnn512
}
