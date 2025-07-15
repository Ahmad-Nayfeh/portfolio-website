---
title: "🏆 منصة ماراثون القراءة: هندسة نظام تحليلي متكامل لتحفيز القراءة"
slug: "reading-marathon-app"
lang: ar
date: "2025-07-13"
coverImage: "/images/projects/reading-marathon-app.jpg"
tags: ["هندسة البيانات", "تحليل البيانات", "Streamlit", "Firebase", "Pandas", "Plotly", "بايثون متكامل", "Google Cloud", "API Integration"]
excerpt: "ابتكار وتطوير منظومة سحابية شاملة (End-to-End System) تستخدم هندسة البيانات والتحليلات المتقدمة لتحويل تجربة القراءة الجماعية إلى رحلة تفاعلية ومحفزة. المنصة تعالج التحديات الإدارية والتحفيزية من خلال خط بيانات آلي، ونظام ألعاب ذكي، ولوحات تحكم متطورة."
category: "تطبيقات وتحليلات البيانات"
githubLink: "https://github.com/Ahmad-Nayfeh/Reading-Tracker-Dashboard-Cloud"
liveDemoUrl: "https://reading-marathon.streamlit.app/"
featured: true
---
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
    .project-container-custom {
        font-family: 'Tajawal', sans-serif;
        line-height: 1.8;
        color: hsl(var(--foreground) / 0.9);
    }
    .section-header-custom {
        text-align: center;
        margin-top: 4rem;
        margin-bottom: 2.5rem;
        position: relative;
    }
    .section-header-custom h2 {
        font-size: 2.25rem;
        font-weight: 800;
        color: hsl(var(--primary));
        margin-bottom: 0.5rem;
    }
    .section-header-custom .subtitle-custom {
        font-size: 1.1rem;
        color: hsl(var(--muted-foreground));
        max-width: 600px;
        margin: 0 auto;
    }
    .section-header-custom::after {
        content: '';
        display: block;
        width: 80px;
        height: 3px;
        background: hsl(var(--primary) / 0.5);
        border-radius: 2px;
        margin: 1.5rem auto 0;
    }
    .cta-buttons-custom {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin: 2rem 0 3rem;
    }
    .cta-buttons-custom a {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 700;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    .cta-button-primary {
        background-color: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
    }
    .cta-button-primary:hover {
        background-color: hsl(var(--primary) / 0.85);
        transform: translateY(-3px);
        box-shadow: 0 4px 15px hsl(var(--primary) / 0.2);
    }
    .cta-button-secondary {
        background-color: hsl(var(--card));
        color: hsl(var(--primary));
        border-color: hsl(var(--primary));
    }
    .cta-button-secondary:hover {
        background-color: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
        transform: translateY(-3px);
    }
    .two-col-layout-custom {
        display: grid;
        grid-template-columns: 1fr;
        gap: 2.5rem;
        margin-top: 2rem;
        align-items: start;
    }
    @media (min-width: 768px) {
        .two-col-layout-custom {
            grid-template-columns: 1fr 1fr;
        }
    }
    .col-box-custom {
        padding: 2rem;
        border-radius: 1rem;
        background: hsl(var(--background));
        border: 1px solid hsl(var(--border));
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.03), 0 2px 4px -2px rgba(0,0,0,0.03);
    }
    .col-box-custom h3 {
        font-size: 1.5rem;
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: hsl(var(--foreground));
        font-weight: 700;
    }
    .col-box-custom ul {
        padding-right: 1.5rem;
        margin: 0;
        list-style-type: '✓ ';
    }
    .col-box-custom li {
        margin-bottom: 0.75rem;
        padding-right: 0.5rem;
    }
    .architecture-diagram-custom {
        margin: 3rem auto;
        padding: 1rem;
        background: hsl(var(--muted) / 0.4);
        border-radius: 1rem;
        border: 1px solid hsl(var(--border));
        text-align: center;
    }
    .architecture-diagram-custom img {
        border-radius: 0.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .architecture-diagram-custom figcaption {
        margin-top: 1rem;
        font-style: italic;
        color: hsl(var(--muted-foreground));
    }
    .features-grid-custom {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin-top: 2rem;
    }
    .feature-card-custom {
        background: hsl(var(--card));
        border: 1px solid hsl(var(--border));
        border-radius: 12px;
        padding: 1.75rem;
        text-align: right;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.05);
    }
    .feature-card-custom:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.07), 0 4px 6px -4px rgba(0,0,0,0.07);
        border-color: hsl(var(--primary) / 0.5);
    }
    .feature-card-custom .icon-custom {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: hsl(var(--primary));
        line-height: 1;
    }
    .feature-card-custom h3 {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0;
        color: hsl(var(--foreground));
    }
    .feature-card-custom p {
        font-size: 0.95rem;
        color: hsl(var(--muted-foreground));
        line-height: 1.7;
    }
    .tech-stack-custom {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 0.75rem;
        margin-top: 2rem;
    }
    .tech-pill-custom {
        background-color: hsl(var(--secondary));
        color: hsl(var(--secondary-foreground));
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid hsl(var(--border));
        transition: all 0.2s ease-in-out;
    }
    .tech-pill-custom:hover {
        background-color: hsl(var(--primary));
        color: hsl(var(--primary-foreground));
        border-color: hsl(var(--primary));
        transform: scale(1.05);
    }
</style>
<div class="project-container-custom" dir="rtl">
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 800; margin-bottom: 1rem;">🏆 منصة ماراثون القراءة</h1>
        <p style="font-size: 1.25rem; max-width: 750px; margin: 0 auto; color: hsl(var(--muted-foreground));">
            هندسة نظام تحليلي متكامل يستخدم قوة البيانات لتحويل تجربة القراءة الجماعية إلى رحلة تفاعلية ومحفزة.
        </p>
    </div>
    <div class="cta-buttons-custom">
        <a href="https://reading-marathon.streamlit.app/" target="_blank" rel="noopener noreferrer" class="cta-button-primary">
            <span>🚀</span>
            <span>جرب التطبيق مباشرة</span>
        </a>
        <a href="https://github.com/Ahmad-Nayfeh/Reading-Tracker-Dashboard-Cloud" target="_blank" rel="noopener noreferrer" class="cta-button-secondary">
            <span>📁</span>
            <span>الكود المصدري</span>
        </a>
    </div>
    <div style="text-align: center; margin-top: -2rem; margin-bottom: 3rem;">
        <a href="/privacy" style="font-size: 0.9rem; text-decoration: underline;">
            اطلع على سياسة الخصوصية الخاصة بالتطبيق
        </a>
    </div>
    <div style="text-align: center; margin-top: -2rem; margin-bottom: 3rem;">
        <a href="/terms" style="font-size: 0.9rem; text-decoration: underline;">
            اطلع على شروط الخدمة الخاصة بالتطبيق
        </a>
    </div>
    <figure class="architecture-diagram-custom">
        <img src="/images/projects/reading-marathon-app/dashboard.png" alt="لقطة شاملة للوحة التحكم في منصة ماراثون القراءة" />
        <figcaption>لوحة التحكم الرئيسية التي توفر رؤى شاملة لأداء المجموعة.</figcaption>
    </figure>
    <div class="section-header-custom">
        <h2>من الفكرة إلى الواقع</h2>
        <p class="subtitle-custom">كيف حولنا تحديات القراءة العربية إلى فرصة للابتكار الهندسي.</p>
    </div>
    <div class="two-col-layout-custom">
        <div class="col-box-custom">
            <h3><span>🎯</span> المشكلة الحقيقية</h3>
            <p>تنطلق فكرة المشروع من تحديات حقيقية تواجه مجموعات القراءة:</p>
            <ul>
                <li><strong>العبء الإداري:</strong> استهلاك وقت المشرفين في مهام يدوية متكررة.</li>
                <li><strong>تآكل الحماس:</strong> غياب آليات التحفيز المرئية والمستمرة.</li>
                <li><strong>قرارات عشوائية:</strong> الاعتماد على الانطباعات بدلاً من البيانات الدقيقة.</li>
            </ul>
        </div>
        <div class="col-box-custom">
            <h3><span>💡</span> رؤيتنا للحل</h3>
            <p>لم نكتفِ ببناء تطبيق، بل صممنا نظامًا بيئيًا متكاملًا يرتكز على:</p>
            <ul>
                <li><strong>الأتمتة الكاملة:</strong> خط بيانات آلي يربط بين جمع البيانات وعرضها بسلاسة.</li>
                <li><strong>التحفيز الذكي:</strong> نظام ألعاب يكافئ الالتزام والإنجاز بفعالية.</li>
                <li><strong>الرؤى العميقة:</strong> تحليلات متعددة المستويات لدعم اتخاذ قرارات مدروسة.</li>
            </ul>
        </div>
    </div>
    <div class="section-header-custom">
        <h2>حلول هندسية مبتكرة</h2>
        <p class="subtitle-custom">استعراض للميزات الرئيسية التي تمثل حلولاً هندسية لمشاكل حقيقية.</p>
    </div>
    <div class="features-grid-custom">
        <div class="feature-card-custom">
            <div class="icon-custom">🔐</div>
            <h3>بنية سحابية آمنة</h3>
            <p>باستخدام Firestore، تم تصميم بنية بيانات معزولة لكل مشرف، مع نظام مصادقة آمن عبر Google OAuth 2.0 لضمان خصوصية البيانات وأمانها.</p>
        </div>
        <div class="feature-card-custom">
            <div class="icon-custom">⚡</div>
            <h3>خط بيانات آلي</h3>
            <p>منظومة مؤتمتة بالكامل تبدأ من نماذج جوجل، مرورًا بالمعالجة عبر Pandas، وصولًا إلى التخزين في Firestore، مما يلغي أي تدخل يدوي.</p>
        </div>
        <div class="feature-card-custom">
            <div class="icon-custom">📊</div>
            <h3>محرك تحليلات ثلاثي</h3>
            <p>يقدم رؤى على 3 مستويات: نظرة شاملة للماراثون، تحليل تكتيكي لكل تحدي، وملف شخصي تحليلي عميق لكل قارئ على حدة.</p>
        </div>
        <div class="feature-card-custom">
            <div class="icon-custom">🎮</div>
            <h3>نظام ألعاب ذكي</h3>
            <p>نظام نقاط وأوسمة ديناميكي ومصمم بعناية لمكافأة الالتزام الجماعي والاستمرارية الفردية، مما يعزز الحماس وروح المنافسة.</p>
        </div>
        <div class="feature-card-custom">
            <div class="icon-custom">⚙️</div>
            <h3>أدوات إدارية فائقة</h3>
            <p>محرر سجلات ذكي يتكامل مع Google Sheets API وميزة تعطيل الأعضاء التي تتفاعل مع Google Forms API لضمان إدارة سلسة وفعالة.</p>
        </div>
        <div class="feature-card-custom">
            <div class="icon-custom">📄</div>
            <h3>مولد تقارير PDF</h3>
            <p>وحدة مخصصة باستخدام FPDF2 تتغلب على تحديات اللغة العربية، لتحويل أي لوحة تحكم إلى تقرير PDF احترافي بنقرة زر.</p>
        </div>
    </div>
    <div class="section-header-custom">
        <h2>الهيكل المعماري للنظام</h2>
        <p class="subtitle-custom">نظرة على كيفية ترابط مكونات النظام لتقديم تجربة سلسة ومتكاملة.</p>
    </div>
    <figure class="architecture-diagram-custom">
        <img src="/images/projects/reading-marathon-app/architecture.png" alt="الهيكل المعماري لمنصة ماراثون القراءة" />
        <figcaption>خط تدفق البيانات من الجمع عبر نماذج جوجل إلى التحليل والعرض في تطبيق ستريمليت.</figcaption>
    </figure>
    <div class="section-header-custom">
        <h2>حزمة التقنيات</h2>
        <p class="subtitle-custom">الأدوات والأطر التي شكلت العمود الفقري لهذا المشروع.</p>
    </div>
    <div class="tech-stack-custom">
        <span class="tech-pill-custom">Python</span>
        <span class="tech-pill-custom">Streamlit</span>
        <span class="tech-pill-custom">Firebase</span>
        <span class="tech-pill-custom">Pandas & NumPy</span>
        <span class="tech-pill-custom">Plotly</span>
        <span class="tech-pill-custom">Google Workspace API</span>
        <span class="tech-pill-custom">Google OAuth</span>
        <span class="tech-pill-custom">FPDF2</span>
    </div>
</div>