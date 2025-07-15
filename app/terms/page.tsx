// File: app/terms/page.tsx

import { FileText, Shield, User, Power, AlertTriangle, RefreshCw } from 'lucide-react';
import type { Metadata } from 'next';

// Add metadata for better SEO
export const metadata: Metadata = {
  title: 'شروط الخدمة | ماراثون القراءة',
  description: 'شروط الخدمة الخاصة باستخدام تطبيق "ماراثون القراءة".',
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


export default function TermsOfServicePage() {
  return (
    // Main container with RTL direction and custom font
    <div dir="rtl" className="bg-background font-sans">
      <div className="container mx-auto max-w-4xl px-4 py-12 md:py-20">
        
        {/* Page Header */}
        <header className="text-center mb-12">
          <div className="inline-block bg-primary/10 p-4 rounded-full mb-4">
            <Shield className="h-10 w-10 text-primary" />
          </div>
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight text-primary">
            شروط الخدمة لتطبيق "ماراثون القراءة"
          </h1>
          <p className="mt-4 text-lg text-muted-foreground">
            باستخدامك لخدماتنا، فإنك توافق على الالتزام بهذه الشروط.
          </p>
          <p className="mt-2 text-sm text-gray-500">آخر تحديث: 15 يوليو 2025</p>
        </header>

        {/* Introduction Section */}
        <div className="prose prose-lg dark:prose-invert max-w-none text-right mb-12 text-lg leading-relaxed">
          <p>
            مرحباً بك في "ماراثون القراءة" ("التطبيق"). باستخدامك لخدماتنا، فإنك توافق على الالتزام بهذه الشروط والأحكام. يرجى قراءتها بعناية.
          </p>
        </div>

        {/* Use of Service Section */}
        <InfoCard icon={<FileText size={24} />} title="1. استخدام الخدمة">
          <ul className="list-none space-y-2">
            <li>- يُقدم التطبيق "كما هو"، وهو مصمم لمساعدتك على إدارة وتتبع تحديات القراءة لمجموعتك.</li>
            <li>- أنت توافق على استخدام التطبيق للأغراض المشروعة فقط وبما لا ينتهك حقوق الآخرين.</li>
          </ul>
        </InfoCard>

        {/* Accounts and Responsibility Section */}
        <InfoCard icon={<User size={24} />} title="2. الحسابات والمسؤولية">
            <ul className="list-none space-y-2">
                <li>- أنت مسؤول عن الحفاظ على أمان حساب جوجل الخاص بك الذي تستخدمه للوصول إلى التطبيق.</li>
                <li>- أنت مسؤول بشكل كامل عن جميع الأنشطة التي تحدث ضمن مساحة العمل الخاصة بك، بما في ذلك البيانات التي تتم مزامنتها من Google Sheets و Google Forms.</li>
                <li>- يلتزم التطبيق بالحفاظ على خصوصية بياناتك وفقاً لـ "سياسة الخصوصية" الخاصة بنا.</li>
            </ul>
        </InfoCard>

        {/* Disclaimer Section */}
        <InfoCard icon={<AlertTriangle size={24} />} title="3. إخلاء المسؤولية">
            <ul className="list-none space-y-2">
                <li>- نحن نسعى لتقديم خدمة مستقرة وموثوقة، ولكننا لا نضمن أن الخدمة ستكون خالية من الأخطاء أو الانقطاعات.</li>
                <li>- نحن غير مسؤولين عن أي فقدان للبيانات قد يحدث نتيجة خطأ في خدمات جوجل (Google Sheets, Forms) أو نتيجة خطأ من المستخدم.</li>
            </ul>
        </InfoCard>

        {/* Termination of Service Section */}
        <InfoCard icon={<Power size={24} />} title="4. إنهاء الخدمة">
            <ul className="list-none space-y-2">
                <li>- يحق لنا تعليق أو إنهاء وصولك إلى الخدمة في حال اكتشاف أي نشاط ينتهك هذه الشروط.</li>
                <li>- يمكنك التوقف عن استخدام الخدمة في أي وقت.</li>
            </ul>
        </InfoCard>

        {/* Changes to Terms Section */}
        <InfoCard icon={<RefreshCw size={24} />} title="5. تغييرات على الشروط">
          <p>
            قد نقوم بتعديل هذه الشروط من وقت لآخر. سيتم نشر أي تغييرات على هذه الصفحة. استمرارك في استخدام التطبيق بعد التغييرات يعني موافقتك عليها.
          </p>
        </InfoCard>

      </div>
    </div>
  );
}
