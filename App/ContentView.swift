
import SwiftUI
import UIKit
import PhotosUI
import Photos
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var sourcePickerItem: PhotosPickerItem?
    @State private var targetPickerItem: PhotosPickerItem?

    @State private var sourceImage: UIImage?
    @State private var targetImage: UIImage?
    @State private var outputImage: UIImage?

    @State private var modelNames: [String] = []
    @State private var selectedSwapModel: FaceFusionSwapModel = .hyperswap1a256
    @State private var showModelImporter = false

    @State private var alertTitle = ""
    @State private var alertMessage = ""
    @State private var showAlert = false

    @State private var isCheckingModels = false
    @State private var isSwapping = false

    @State private var modelDiagnosticReport = ""
    @State private var processingStatus = ""

    private let columns = [
        GridItem(.flexible()),
        GridItem(.flexible())
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    header
                    privacyNotice

                    LazyVGrid(columns: columns, spacing: 12) {
                        imagePickerCard(
                            title: "صورة الوجه المصدر",
                            subtitle: "الوجه الذي تريد استخدامه",
                            image: sourceImage,
                            selection: $sourcePickerItem
                        )

                        imagePickerCard(
                            title: "الصورة الهدف",
                            subtitle: "الصورة التي ستتغير",
                            image: targetImage,
                            selection: $targetPickerItem
                        )
                    }

                    modelSelector

                    Button {
                        showModelImporter = true
                    } label: {
                        Label(
                            "استيراد ملفات نموذج ONNX",
                            systemImage: "shippingbox"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)

                    modelStatus

                    Button {
                        runModelDiagnostics()
                    } label: {
                        Label(
                            isCheckingModels
                                ? "جارٍ فحص النماذج…"
                                : "فحص اتصال النماذج",
                            systemImage: "cpu"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.large)
                    .disabled(isCheckingModels || modelNames.isEmpty)

                    if !modelDiagnosticReport.isEmpty {
                        Text(modelDiagnosticReport)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                            .frame(
                                maxWidth: .infinity,
                                alignment: .leading
                            )
                            .padding(12)
                            .background(
                                Color(uiColor: .secondarySystemGroupedBackground),
                                in: RoundedRectangle(cornerRadius: 12)
                            )
                    }

                    Button {
                        runFaceSwap()
                    } label: {
                        Label(
                            isSwapping
                                ? "جارٍ تبديل الوجه…"
                                : "تبديل الوجه",
                            systemImage: "person.crop.square"
                        )
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(
                        isSwapping ||
                        sourceImage == nil ||
                        targetImage == nil
                    )

                    if isSwapping {
                        ProgressView(
                            "جارٍ تشغيل نماذج الوجه محليًا…"
                        )
                        .frame(maxWidth: .infinity)
                    }

                    if !processingStatus.isEmpty {
                        Text(processingStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .fixedSize(
                                horizontal: false,
                                vertical: true
                            )
                    }

                    if let outputImage {
                        resultSection(outputImage)
                    }

                    Text(
                        "تحليل الوجه يستخدم نماذج FaceFusion الأصلية عبر ONNX Runtime: " +
                        "yoloface_8n للكشف و2dfan4 لمعالم الوجه، وليس Vision. " +
                        "استورد ملفات النماذج يدويًا؛ لا يوجد تنزيل تلقائي. تتم معالجة صورة ثابتة وأكبر وجه مكتشف فقط."
                    )
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .fixedSize(
                        horizontal: false,
                        vertical: true
                    )

                    footer
                }
                .padding(16)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle("FaceFusion Local")
            .navigationBarTitleDisplayMode(.inline)
            .onAppear {
                refreshModelNames()
            }
            .onChange(of: sourcePickerItem) { newValue in
                Task {
                    await loadImage(from: newValue, isSource: true)
                }
            }
            .onChange(of: targetPickerItem) { newValue in
                Task {
                    await loadImage(from: newValue, isSource: false)
                }
            }
            .fileImporter(
                isPresented: $showModelImporter,
                allowedContentTypes: [.data],
                allowsMultipleSelection: true,
                onCompletion: handleModelImport
            )
            .alert(alertTitle, isPresented: $showAlert) {
                Button("حسنًا", role: .cancel) {}
            } message: {
                Text(alertMessage)
            }
        }
        .environment(\.layoutDirection, .rightToLeft)
    }

    // MARK: - Interface

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 12) {
                Image(
                    systemName: "person.crop.square.filled.and.at.rectangle"
                )
                .font(.system(size: 28, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 56, height: 56)
                .background(
                    LinearGradient(
                        colors: [.indigo, .purple],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    in: RoundedRectangle(cornerRadius: 16)
                )

                VStack(alignment: .leading, spacing: 4) {
                    Text("FaceFusion Local")
                        .font(.title2.bold())

                    Text("تبديل الوجه على الجهاز")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }

            Text(
                "تتم معالجة الصور محليًا على الآيفون دون إرسالها إلى خادم."
            )
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var privacyNotice: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "lock.shield.fill")
                .font(.title3)
                .foregroundStyle(.green)

            VStack(alignment: .leading, spacing: 4) {
                Text("الخصوصية أولًا")
                    .font(.headline)

                Text(
                    "المحرك الجديد لا يحتاج إلى اتصال بخدمة معالجة خارجية. " +
                    "النماذج والصور تُستخدم داخل مساحة التطبيق وعلى الجهاز."
                )
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            Color(uiColor: .secondarySystemGroupedBackground),
            in: RoundedRectangle(cornerRadius: 16)
        )
    }

    private func imagePickerCard(
        title: String,
        subtitle: String,
        image: UIImage?,
        selection: Binding<PhotosPickerItem?>
    ) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            PhotosPicker(
                selection: selection,
                matching: .images,
                photoLibrary: .shared()
            ) {
                ZStack {
                    RoundedRectangle(cornerRadius: 14)
                        .fill(
                            Color(uiColor: .tertiarySystemGroupedBackground)
                        )

                    if let image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .clipShape(
                                RoundedRectangle(cornerRadius: 14)
                            )
                    } else {
                        VStack(spacing: 8) {
                            Image(systemName: "plus.viewfinder")
                                .font(.system(size: 30))

                            Text("اختيار صورة")
                                .font(.subheadline.weight(.medium))
                        }
                        .foregroundStyle(.secondary)
                    }
                }
                .frame(height: 170)
                .clipShape(RoundedRectangle(cornerRadius: 14))
            }
            .buttonStyle(.plain)

            Text(title)
                .font(.headline)
                .lineLimit(2)

            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(
            Color(uiColor: .secondarySystemGroupedBackground),
            in: RoundedRectangle(cornerRadius: 18)
        )
    }

    private var modelSelector: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("نموذج تبديل الوجه من FaceFusion")
                .font(.headline)

            Picker("نموذج تبديل الوجه", selection: $selectedSwapModel) {
                ForEach(FaceFusionSwapModel.allCases) { model in
                    Text(model.displayName)
                        .tag(model)
                        .disabled(!model.isSupportedByCurrentEngine)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(
                Color(uiColor: .tertiarySystemGroupedBackground),
                in: RoundedRectangle(cornerRadius: 12)
            )

            Text("الملفات المطلوبة: " + selectedSwapModel.requiredModelFiles.joined(separator: "، "))
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)

            if let note = selectedSwapModel.supportNote {
                Label(note, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            Color(uiColor: .secondarySystemGroupedBackground),
            in: RoundedRectangle(cornerRadius: 16)
        )
    }

    private var modelStatus: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("النماذج المحلية")
                    .font(.headline)

                Spacer()

                Text("\(modelNames.count) ملف")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            if modelNames.isEmpty {
                Label(
                    "لم يتم استيراد ملفات نماذج",
                    systemImage: "exclamationmark.circle"
                )
                .font(.subheadline)
                .foregroundStyle(.secondary)
            } else {
                ForEach(modelNames, id: \.self) { name in
                    Label(name, systemImage: "doc")
                        .font(.caption)
                        .lineLimit(1)
                }
            }

            let missing = selectedSwapModel.requiredModelFiles.filter { !modelNames.contains($0) }
            if missing.isEmpty {
                Label("ملفات النموذج المختار موجودة", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                Text("ملفات ناقصة للنموذج المختار: " + missing.joined(separator: "، "))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            Color(uiColor: .secondarySystemGroupedBackground),
            in: RoundedRectangle(cornerRadius: 16)
        )
    }

    private func resultSection(_ image: UIImage) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("النتيجة")
                .font(.headline)

            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxWidth: .infinity)
                .frame(maxHeight: 420)
                .clipShape(RoundedRectangle(cornerRadius: 14))

            Button {
                saveResultToPhotos(image)
            } label: {
                Label(
                    "حفظ النتيجة في الصور",
                    systemImage: "square.and.arrow.down"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .controlSize(.large)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            Color(uiColor: .secondarySystemGroupedBackground),
            in: RoundedRectangle(cornerRadius: 16)
        )
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 5) {
            Divider()

            Text(
                "استخدم صورًا تملك حق استخدامها، ولا تبدّل وجه شخص حقيقي دون موافقته."
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
        .padding(.top, 6)
    }

    // MARK: - Load images

    @MainActor
    private func loadImage(
        from item: PhotosPickerItem?,
        isSource: Bool
    ) async {
        guard let item else { return }

        do {
            guard let data = try await item.loadTransferable(type: Data.self),
                  let image = UIImage(data: data) else {
                showMessage(
                    "تعذر فتح الصورة",
                    "جرّب صورة أخرى من مكتبة الصور."
                )
                return
            }

            if isSource {
                sourceImage = image
            } else {
                targetImage = image
            }

            outputImage = nil
            processingStatus = ""
        } catch {
            showMessage(
                "تعذر فتح الصورة",
                error.localizedDescription
            )
        }
    }

    // MARK: - Run face swap

    private func runFaceSwap() {
        guard let source = sourceImage,
              let target = targetImage else {
            showMessage(
                "اختر الصورتين",
                "يجب اختيار صورة المصدر والصورة الهدف أولًا."
            )
            return
        }

        let folder: URL

        do {
            folder = try modelFolderURL()
        } catch {
            showMessage(
                "تعذر الوصول إلى النماذج",
                error.localizedDescription
            )
            return
        }

        guard selectedSwapModel.isSupportedByCurrentEngine else {
            showMessage(
                "النموذج غير جاهز",
                selectedSwapModel.supportNote ?? "هذا النموذج غير مدعوم في هذا الإصدار."
            )
            return
        }

        let requiredModels = selectedSwapModel.requiredModelFiles

        for name in requiredModels {
            let url = folder.appendingPathComponent(name)

            guard FileManager.default.fileExists(atPath: url.path) else {
                showMessage(
                    "نموذج ناقص",
                    "استورد الملف \(name) ثم جرّب مرة أخرى."
                )
                return
            }
        }

        isSwapping = true
        outputImage = nil
        processingStatus = "جارٍ تجهيز الوجه وتشغيل \(selectedSwapModel.displayName) محليًا…"

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let result = try FaceSwapEngine.swap(
                    source: source,
                    target: target,
                    modelDirectory: folder,
                    selectedModel: selectedSwapModel
                )

                DispatchQueue.main.async {
                    outputImage = result
                    isSwapping = false
                    processingStatus =
                        "اكتملت المعالجة. راجع النتيجة قبل حفظها."
                }
            } catch {
                DispatchQueue.main.async {
                    isSwapping = false
                    processingStatus = ""
                    showMessage(
                        "تعذر تبديل الوجه",
                        error.localizedDescription
                    )
                }
            }
        }
    }

    // MARK: - Save result

    private func saveResultToPhotos(_ image: UIImage) {
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    showMessage(
                        "تعذر الحفظ",
                        "اسمح للتطبيق بإضافة الصور إلى مكتبة الصور في إعدادات الآيفون."
                    )
                }
                return
            }

            PHPhotoLibrary.shared().performChanges {
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            } completionHandler: { success, error in
                DispatchQueue.main.async {
                    if success {
                        showMessage(
                            "تم الحفظ",
                            "تم حفظ الصورة الناتجة في مكتبة الصور."
                        )
                    } else {
                        showMessage(
                            "فشل الحفظ",
                            error?.localizedDescription ??
                            "تعذر حفظ الصورة."
                        )
                    }
                }
            }
        }
    }

    // MARK: - Import models

    private func handleModelImport(
        _ result: Result<[URL], Error>
    ) {
        switch result {
        case .failure(let error):
            showMessage(
                "فشل استيراد الملفات",
                error.localizedDescription
            )

        case .success(let urls):
            do {
                let folder = try modelFolderURL()
                var imported: [String] = []

                for url in urls {
                    let granted = url.startAccessingSecurityScopedResource()
                    defer {
                        if granted {
                            url.stopAccessingSecurityScopedResource()
                        }
                    }

                    guard url.pathExtension.lowercased() == "onnx" else {
                        continue
                    }

                    let destination = folder.appendingPathComponent(
                        url.lastPathComponent
                    )

                    if FileManager.default.fileExists(
                        atPath: destination.path
                    ) {
                        try FileManager.default.removeItem(at: destination)
                    }

                    try FileManager.default.copyItem(
                        at: url,
                        to: destination
                    )

                    imported.append(url.lastPathComponent)
                }

                refreshModelNames()

                showMessage(
                    "اكتمل الاستيراد",
                    "تم استيراد \(imported.count) ملف ONNX إلى مساحة التطبيق المحلية."
                )
            } catch {
                showMessage(
                    "تعذر نسخ الملفات",
                    error.localizedDescription
                )
            }
        }
    }

    private func modelFolderURL() throws -> URL {
        let documents = try FileManager.default.url(
            for: .documentDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        )

        let folder = documents.appendingPathComponent(
            "Models",
            isDirectory: true
        )

        try FileManager.default.createDirectory(
            at: folder,
            withIntermediateDirectories: true
        )

        return folder
    }

    private func refreshModelNames() {
        do {
            let folder = try modelFolderURL()

            let urls = try FileManager.default.contentsOfDirectory(
                at: folder,
                includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]
            )

            modelNames = urls
                .filter { $0.pathExtension.lowercased() == "onnx" }
                .map { $0.lastPathComponent }
                .sorted()
        } catch {
            showMessage(
                "تعذر قراءة مجلد النماذج",
                error.localizedDescription
            )
        }
    }

    // MARK: - Diagnostics

    private func runModelDiagnostics() {
        guard !isCheckingModels else { return }

        do {
            let folder = try modelFolderURL()
            isCheckingModels = true
            modelDiagnosticReport = "جارٍ اختبار فتح النماذج محليًا…"

            DispatchQueue.global(qos: .userInitiated).async {
                let report = ModelPipelineInspector.inspect(
                    directory: folder,
                    selectedModel: selectedSwapModel
                )

                DispatchQueue.main.async {
                    modelDiagnosticReport = report
                    isCheckingModels = false
                }
            }
        } catch {
            showMessage(
                "تعذر الوصول إلى مجلد النماذج",
                error.localizedDescription
            )
        }
    }

    private func showMessage(
        _ title: String,
        _ message: String
    ) {
        alertTitle = title
        alertMessage = message
        showAlert = true
    }
}
