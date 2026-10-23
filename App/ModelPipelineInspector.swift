import Foundation
import OnnxRuntimeBindings

/// Opens manually imported ONNX files and displays their graph signatures.
/// This is a compatibility diagnostic, not a proof of correct face swapping.
enum ModelPipelineInspector {
    static func inspect(
        directory: URL,
        selectedModel: FaceFusionSwapModel
    ) -> String {
        var lines = [
            "تقرير توافق ONNX Runtime",
            "المجلد: Documents/Models",
            "النموذج المختار: \(selectedModel.displayName)",
            "المطلوب لهذا النموذج: \(selectedModel.requiredModelFiles.joined(separator: ", "))",
            "----------------------------------------"
        ]

        let missing = selectedModel.requiredModelFiles.filter {
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        if missing.isEmpty {
            lines.append("✓ كل الملفات المطلوبة للنموذج المختار موجودة.")
        } else {
            lines.append("✗ ملفات ناقصة للنموذج المختار: \(missing.joined(separator: ", "))")
        }

        if !selectedModel.isSupportedByCurrentEngine, let note = selectedModel.supportNote {
            lines.append("! ملاحظة: \(note)")
        }

        let env: ORTEnv
        let options: ORTSessionOptions
        do {
            env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)
            options = try ORTSessionOptions()
            try options.setIntraOpNumThreads(2)
        } catch {
            lines.append("✗ تعذرت تهيئة ONNX Runtime: \(error.localizedDescription)")
            return lines.joined(separator: "\n")
        }

        let files: [URL]
        do {
            files = try FileManager.default.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles]
            )
            .filter { $0.pathExtension.lowercased() == "onnx" }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
        } catch {
            lines.append("✗ تعذر قراءة مجلد النماذج: \(error.localizedDescription)")
            return lines.joined(separator: "\n")
        }

        guard !files.isEmpty else {
            lines.append("لا توجد ملفات ONNX مستوردة.")
            return lines.joined(separator: "\n")
        }

        lines.append("")
        lines.append("توقيعات النماذج المستوردة")
        for fileURL in files {
            let filename = fileURL.lastPathComponent
            do {
                let details = try autoreleasepool { () throws -> (String, String, String) in
                    let session = try ORTSession(
                        env: env,
                        modelPath: fileURL.path,
                        sessionOptions: options
                    )
                    let inputs = try session.inputNames().joined(separator: ", ")
                    let outputs = try session.outputNames().joined(separator: ", ")
                    let bytes = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
                    let size = String(format: "%.1f MB", Double(bytes) / 1_048_576.0)
                    return (inputs, outputs, size)
                }
                lines.append("✓ \(filename) (\(details.2))")
                lines.append("  Inputs: \(details.0)")
                lines.append("  Outputs: \(details.1)")
            } catch {
                lines.append("✗ \(filename)")
                lines.append("  السبب: \(error.localizedDescription)")
            }
        }

        lines.append("")
        lines.append("نجاح فتح الجلسة لا يثبت صحة المداخل أو جودة صورة تبديل الوجه.")
        lines.append("لا ينزّل التطبيق أي نموذج تلقائيًا؛ هذه الفحوصات محلية.")
        return lines.joined(separator: "\n")
    }
}
