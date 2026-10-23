# FaceFusion Local for iPhone — نسخة تجريبية

هذا المشروع يعيد تنفيذ جزء من خط معالجة الصور في FaceFusion الأصلي باستخدام SwiftUI وONNX Runtime iOS. صُمم للعمل محليًا على الصور الثابتة، مع استيراد ملفات النماذج يدويًا. **لا توجد ميزة تنزيل تلقائي للنماذج ولا رفع للصور إلى خادم.**

## ما الذي يحتويه المشروع؟

- اختيار صورة المصدر وصورة الهدف من مكتبة الصور.
- استيراد ملفات `.onnx` عبر تطبيق Files ونسخها إلى `Documents/Models`.
- قائمة بأسماء نماذج تبديل الوجه الموجودة في FaceFusion الرسمي: AlphaFace، BlendSwap، Ghost 1/2/3، HifiFace، HyperSwap 1A/1B/1C، InSwapper 128/FP16، SimSwap 256، SimSwap unofficial 512، وUniFace.
- تحليل وجه مبني على نماذج FaceFusion الأصلية عبر ONNX Runtime: `yoloface_8n.onnx` للكشف والنقاط الخمس، و`2dfan4.onnx` لمعالم الوجه الـ68. لا يستخدم التطبيق Apple Vision لتحليل الوجه.
- مسارات وتجهيزات مختلفة للنماذج التي تستخدم embedding، والنماذج التي تأخذ صورة المصدر مباشرة، ومحوّلات الهوية المطلوبة لعائلات Ghost وHifiFace وSimSwap.
- أداة تشخيص تفتح كل ملف ONNX مستورد وتعرض أسماء المداخل والمخارج.
- حفظ النتيجة في مكتبة الصور.

## الملفات التي يجب استيرادها

بالإضافة إلى نموذج تبديل الوجه الذي تختاره، يلزم استيراد ملفات تحليل الوجه الأصلية التالية يدويًا:

- `yoloface_8n.onnx` — كاشف FaceFusion الرسمي الافتراضي، ينتج صندوق الوجه وخمس نقاط محاذاة.
- `2dfan4.onnx` — محدد المعالم 68 نقطة المستخدم في FaceFusion.
- `arcface_w600k_r50.onnx` — لاستخراج تضمين هوية المصدر.
- ملف النموذج المختار، مثل `hyperswap_1a_256.onnx`.
- لبعض العائلات، محوّل الهوية أيضًا:
  - Ghost: `crossface_ghost.onnx`
  - HifiFace: `crossface_hififace.onnx`
  - SimSwap: `crossface_simswap.onnx`

تُظهر واجهة التطبيق قائمة الملفات المطلوبة للنموذج المحدد. انقل كل الملفات اللازمة من تطبيق Files؛ لا ينزّل التطبيق أي ملفات بنفسه. أسماء الملفات مهمة لأن المحرك يستخدمها للتعرّف على الأصول.

## النماذج والقيود الحالية

- يبدأ الاختيار الافتراضي بـ **HyperSwap 1A 256**.
- يتضمن مسار InSwapper قارئًا محدودًا لمصفوفة `emap` داخل ملف ONNX المستورد، لكنه يحتاج اختبارًا على ملفات InSwapper الأصلية (بما فيها نسخة FP16) قبل الاعتماد عليه.
- لم يتم نقل كل خيارات الكشف في FaceFusion بعد؛ هذا الإصدار يستخدم `yoloface_8n.onnx` و`2dfan4.onnx` كمسار أصلي افتراضي عبر ONNX Runtime. لا توجد أي استدعاءات Apple Vision لكشف الوجه أو تحديد المعالم.
- الأقنعة والدمج يستخدمان قناعًا بيضاويًا ناعمًا أوليًا، وليس كامل نظام الأقنعة في FaceFusion.
- النموذج الذي ينجح في الفتح قد يفشل أثناء الاستدلال إذا اختلفت نسخته أو أنواع بياناته أو أبعاد مداخله ومخارجه. تقرير التشخيص لا يثبت جودة النتيجة.
- لم يُختبر البناء النهائي على iPhone في بيئة التحرير هذه. يجب بناء المشروع ثم تقييمه على الجهاز، مع الانتباه إلى الذاكرة والحرارة والوقت، خصوصًا مع نماذج 512 بكسل.

## المتطلبات

- Windows 10/11 لإنشاء المستودع ورفع المشروع.
- حساب GitHub لاستخدام GitHub Actions.
- عند تثبيت التطبيق على iPhone: أداة توقيع / sideloading متوافقة مع جهازك وحساب Apple.

## البناء من Windows باستخدام GitHub Actions

1. فك ضغط المشروع إلى مجلد مثل `C:\FaceFusion-iOS-Starter`.
2. أنشئ مستودعًا جديدًا على GitHub وارفع محتويات المجلد. لا ترفع صورًا شخصية أو ملفات نماذج لا تملك حق إعادة توزيعها.
3. من PowerShell داخل مجلد المشروع:

```powershell
git init
git add .
git commit -m "Add FaceFusion model selection and local iOS pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/facefusion-ios-local.git
git push -u origin main
```

4. افتح تبويب **Actions** في GitHub، وتأكد من تشغيل **Build unsigned iOS IPA**.
5. بعد نجاح عملية البناء، نزّل الـ artifact `FaceFusionLocal-unsigned-IPA` من صفحة تشغيل الـ workflow.
6. ملف IPA غير الموقّع لا يُثبّت مباشرة. وقّعه بأداة sideloading مناسبة قبل تثبيته.

## التراخيص

الكود الأصلي وأوزان النماذج ليست لها بالضرورة نفس الرخصة. لا يتضمن هذا المستودع أي وزن. قبل الاستخدام أو التوزيع، راجع ترخيص FaceFusion وترخيص كل نموذج ومحوّل embedding بشكل منفصل؛ قد تقيّد بعض الأوزان الاستخدام التجاري أو إعادة التوزيع.

## مراجع

- FaceFusion: https://github.com/facefusion/facefusion
- ONNX Runtime iOS: https://onnxruntime.ai/docs/get-started/with-obj-c.html
- ONNX Runtime Core ML Execution Provider: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
