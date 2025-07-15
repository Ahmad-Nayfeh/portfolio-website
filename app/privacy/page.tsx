// File: app/privacy/page.tsx

import { ShieldCheck, UserCircle, Database, Trash2, FileText, GitBranch } from 'lucide-react';
import type { Metadata } from 'next';

// Add metadata for better SEO
export const metadata: Metadata = {
  title: 'سياسة الخصوصية | ماراثون القراءة',
  description: 'سياسة الخصوصية لتطبيق "ماراثون القراءة"، نوضح فيها كيفية جمع واستخدام وحماية بياناتك.',
};

// A reusable component for section cards to keep the code clean
const InfoCard = ({ icon, title, children }: { icon: React.ReactNode, title: string, children: React.ReactNode }) => (
  <div className="mb-8 rounded-xl border bg-card text-card-foreground shadow-sm transition-all hover:shadow-md">
    <div className="p-6 flex items-start space-x-4 space-x-reverse">
      <div className="flex-shrink-0 mt-1">
        <span className="inline-flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
          {icon}
        </span>
      </div>
      <div>
        <h3 className="text-xl font-bold text-primary mb-2">{title}</h3>
        <div className="text-muted-foreground space-y-3">{children}</div>
      </div>
    </div>
  </div>
);

// A reusable component for list items
const ListItem = ({ children }: { children: React.ReactNode }) => (
  <li className="flex items-start">
    <ShieldCheck className="h-5 w-5 text-green-500 mt-1 ml-3 flex-shrink-0" />
    <span>{children}</span>
  </li>
);

export default function PrivacyPolicy() {
  return (
    // Main container with RTL direction and custom font
    <div dir="rtl" className="bg-background font-sans">
      <div className="container mx-auto max-w-4xl px-4 py-12 md:py-20">
        
        {/* Page Header */}
        <header className="text-center mb-12">
          <div className="inline-block bg-primary/10 p-4 rounded-full mb-4">
            <FileText className="h-10 w-10 text-primary" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-primary">
            سياسة الخصوصية لتطبيق "ماراثون القراءة"
          </h1>
          <p className="mt-4 text-lg text-muted-foreground">
            خصوصيتك هي أولويتنا القصوى.
          </p>
          <p className="mt-2 text-sm text-gray-500">آخر تحديث: 15 يوليو 2025</p>
        </header>

        {/* Introduction Section */}
        <div className="prose prose-lg dark:prose-invert max-w-none text-right mb-12 text-lg leading-relaxed">
          <p>
            أهلاً بك في "ماراثون القراءة". نحن نأخذ خصوصيتك على محمل الجد. توضح هذه السياسة أنواع البيانات التي نجمعها وكيفية استخدامها وحمايتها عند استخدامك لتطبيقنا.
          </p>
        </div>

        {/* Data Collection Section */}
        <InfoCard icon={<UserCircle size={24} />} title="البيانات التي نجمعها">
          <p>
            عندما تقوم بربط حسابك في جوجل مع تطبيق "ماراثون القراءة"، فإننا نطلب صلاحية الوصول إلى المعلومات التالية:
          </p>
          <ul className="space-y-4 mt-4">
            <ListItem>
              <b>معلومات الملف الشخصي الأساسية:</b> مثل اسمك، بريدك الإلكتروني، وصورتك الشخصية. نستخدم هذه المعلومات لإنشاء حسابك وتخصيص تجربتك داخل التطبيق.
            </ListItem>
            <ListItem>
              <b>معرّف جوجل (User ID):</b> نستخدمه كمعرّف فريد لإنشاء مساحة عمل خاصة بك في قاعدة بياناتنا.
            </ListItem>
          </ul>
        </InfoCard>

        {/* How We Use Your Data Section */}
        <InfoCard icon={<GitBranch size={24} />} title="كيف نستخدم بياناتك">
          <p>
            يستخدم تطبيق "ماراثون القراءة" الصلاحيات التي تمنحها له للقيام بالمهام التالية، والتي تحدث <strong>داخل حساب جوجل الخاص بك وتحت سيطرتك الكاملة</strong>:
          </p>
          <ul className="space-y-4 mt-4">
            <ListItem>
              <b>إنشاء وإدارة Google Sheets:</b> يقوم التطبيق بإنشاء جدول بيانات واحد في حسابك ليكون بمثابة قاعدة البيانات الخام التي تُخزن فيها سجلات القراءة التي يدخلها أعضاء فريقك.
            </ListItem>
            <ListItem>
              <b>إنشاء وإدارة Google Forms:</b> يقوم التطبيق بإنشاء نموذج تسجيل واحد في حسابك ليستخدمه أعضاء فريقك لتسجيل قراءاتهم اليومية بسهولة.
            </ListItem>
            <ListItem>
              <b>الوصول إلى Google Drive:</b> يستخدم التطبيق هذه الصلاحية فقط للتمكن من إنشاء وحذف الملفات المذكورة أعلاه (Sheet و Form) المرتبطة بحسابك في التطبيق.
            </ListItem>
          </ul>
          <div className="mt-4 p-4 border-r-4 border-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-md">
            <p className="font-semibold text-amber-800 dark:text-amber-300">
              ملاحظة هامة: التطبيق لا يقوم بقراءة أو تعديل أو حذف أي ملفات أخرى في حسابك على Google Drive غير تلك التي يقوم هو بإنشائها.
            </p>
          </div>
        </InfoCard>

        {/* Data Storage Section */}
        <InfoCard icon={<Database size={24} />} title="تخزين البيانات وحمايتها">
          <p>
            يتم تخزين بياناتك المتعلقة بالتطبيق (مثل إعداداتك، أسماء أعضائك، والتحديات) بشكل آمن في قاعدة بيانات Google Firestore. نحن لا نشارك بياناتك الشخصية مع أي طرف ثالث.
          </p>
        </InfoCard>

        {/* Data Deletion Section */}
        <InfoCard icon={<Trash2 size={24} />} title="حذف البيانات">
          <p>
            لديك السيطرة الكاملة على بياناتك. يمكنك في أي وقت حذف حسابك بالكامل من خلال الذهاب إلى "الإدارة والإعدادات" داخل التطبيق واختيار "حذف الحساب". هذا الإجراء سيقوم بحذف جميع بياناتك من قاعدة بياناتنا وملفات جوجل التي أنشأها التطبيق في حسابك.
          </p>
        </InfoCard>
      </div>
    </div>
  );
}
