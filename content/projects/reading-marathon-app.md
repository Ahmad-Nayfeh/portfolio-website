---
title: "منصة ماراثون القراءة: تطبيق تحليلي متكامل"
slug: "reading-marathon-app"
date: "2025-07-13"
coverImage: "/images/projects/reading-marathon-app.jpg"
tags: ["Data Analysis", "Data Visualization", "Streamlit", "Firebase", "Pandas", "Plotly", "Full-Stack Python", "Google Cloud", "Data Engineering"]
excerpt: "تطبيق ويب متكامل لإدارة وتحليل بيانات مجموعات القراءة، مبني باستخدام Python و Streamlit. يتميز التطبيق بنظام ألعاب (Gamification) ومزامنة سحابية مع Google Sheets وقاعدة بيانات Firestore مع لوحات تحكم تفاعلية وتقارير PDF آلية."
category: "Data Application & Analytics"
githubLink: "https://github.com/Ahmad-Nayfeh/Reading-Tracker-Dashboard-Cloud"
liveDemoUrl: "https://reading-marathon.streamlit.app/"
challenge: "تكمن الصعوبة في إدارة مجموعات القراءة في التحديات المستمرة لمتابعة التزام الأعضاء، والحفاظ على تفاعلهم، والعبء الإداري الكبير على المشرفين. كانت هناك حاجة ماسة لأداة مركزية توفر رؤى قائمة على البيانات وتجعل التجربة ممتعة ومحفزة."
solution: "قمت بتصميم وبناء منصة سحابية شاملة تعمل كنظام متكامل. تستخدم المنصة Google Forms لجمع البيانات بسلاسة، وتزامنها مع قاعدة بيانات Firestore السحابية لكل مستخدم، وتعرضها في لوحات تحكم تفاعلية مبنية بـ Streamlit. تم دمج نظام نقاط وأوسمة ذكي لتحفيز المشاركين، مع أدوات إدارية قوية للمشرفين."
technologies:
  - "Python"
  - "Streamlit"
  - "Firebase (Firestore)"
  - "Pandas & NumPy"
  - "Plotly"
  - "Google Workspace API (Sheets, Forms)"
  - "Google OAuth"
  - "FPDF2 (for PDF Reports)"
features:
  - "بنية سحابية آمنة ومتعددة المستخدمين مع مصادقة جوجل."
  - "خط بيانات آلي (Automated Data Pipeline) من Google Forms إلى التطبيق."
  - "قاعدة بيانات NoSQL (Firestore) لتخزين البيانات بشكل دائم ومنظم."
  - "لوحات تحكم تحليلية تفاعلية (عامة، للتحدي، وللقارئ)."
  - "نظام ألعاب (Gamification) ونقاط وأوسمة لتحفيز المستخدمين."
  - "لوحة تحكم إدارية كاملة (إدارة الأعضاء والتحديات، ومحرر بيانات)."
  - "إنشاء تقارير PDF احترافية باللغة العربية بشكل آلي."
  - "واجهة مستخدم حديثة وجذابة تدعم اللغة العربية بالكامل."
featured: true
---

<div dir="rtl" style="font-family: inherit; color: hsl(var(--foreground)); text-align: right !important;">

  <header style="text-align: center; margin-bottom: 4rem; border-bottom: 1px solid hsl(var(--border)); padding-bottom: 2rem;">
    <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 1rem;">🏃‍♂️ منصة ماراثون القراءة: تطبيق تحليلي متكامل</h1>
    <p style="font-size: 1.25rem; color: hsl(var(--muted-foreground));">
      مشروع شامل يوضح بناء تطبيق بيانات (Data App) من مرحلة الفكرة إلى النشر والإنتاج، باستخدام Python وتقنيات سحابية حديثة.
    </p>
    <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1rem;">
      <a href="https://reading-marathon.streamlit.app/" target="_blank" rel="noopener noreferrer" style="text-decoration: none; background-color: hsl(var(--primary)); color: hsl(var(--primary-foreground)); padding: 0.75rem 1.5rem; border-radius: var(--radius); font-weight: 500; display: inline-flex; align-items: center; gap: 0.5rem; transition: opacity 0.2s;
      --hover-opacity: 0.9;" onmouseover="this.style.opacity = this.style.getPropertyValue('--hover-opacity')" onmouseout="this.style.opacity = 1">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>
        العرض المباشر
      </a>
      <a href="https://github.com/Ahmad-Nayfeh/Reading-Tracker-Dashboard-Cloud" target="_blank" rel="noopener noreferrer" style="text-decoration: none; background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.75rem 1.5rem; border-radius: var(--radius); font-weight: 500; display: inline-flex; align-items: center; gap: 0.5rem; transition: opacity 0.2s;
      --hover-opacity: 0.8;" onmouseover="this.style.opacity = this.style.getPropertyValue('--hover-opacity')" onmouseout="this.style.opacity = 1">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/></svg>
        الكود المصدري
      </a>
    </div>
  </header>

  <section style="margin-bottom: 4rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 2rem;">
    <div style="background-color: hsl(var(--card)); border: 1px solid hsl(var(--border)); border-radius: var(--radius); padding: 1.5rem;">
      <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">التقنيات المستخدمة</h3>
      <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
        <span style="background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Python</span>
        <span style="background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Streamlit</span>
        <span style="background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Firebase</span>
        <span style="background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Pandas</span>
        <span style="background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Plotly</span>
        <span style="background-color: hsl(var(--secondary)); color: hsl(var(--secondary-foreground)); padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem;">Google Cloud</span>
      </div>
    </div>
    <div style="background-color: hsl(var(--card)); border: 1px solid hsl(var(--border)); border-radius: var(--radius); padding: 1.5rem;">
      <h3 style="font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem;">أبرز المميزات</h3>
      <div style="display: flex; flex-direction: column; gap: 0.75rem; text-align: right;">
        <div style="display: flex; align-items: center; gap: 0.75rem;"><span style="color: hsl(var(--primary)); font-weight: bold;">✓</span><span>بنية سحابية آمنة ومتعددة المستخدمين.</span></div>
        <div style="display: flex; align-items: center; gap: 0.75rem;"><span style="color: hsl(var(--primary)); font-weight: bold;">✓</span><span>أتمتة كاملة لخط بيانات (Data Pipeline).</span></div>
        <div style="display: flex; align-items: center; gap: 0.75rem;"><span style="color: hsl(var(--primary)); font-weight: bold;">✓</span><span>نظام ألعاب (Gamification) لتحفيز المستخدمين.</span></div>
        <div style="display: flex; align-items: center; gap: 0.75rem;"><span style="color: hsl(var(--primary)); font-weight: bold;">✓</span><span>إنشاء تقارير PDF آلية باللغة العربية.</span></div>
      </div>
    </div>
  </section>

  <article>
    <section style="margin-bottom: 3rem; text-align: right;">
      <h2 style="font-size: 2rem; font-weight: bold; margin-bottom: 1rem; border-bottom: 2px solid hsl(var(--primary)); padding-bottom: 0.5rem; display: inline-block;">نظرة عامة</h2>
      <p style="line-height: 1.8; font-size: 1.1rem; color: hsl(var(--muted-foreground));">
        "ماراثون القراءة" هو أكثر من مجرد لوحة تحكم؛ إنه منصة متكاملة قمت بتصميمها وتطويرها بالكامل لحل مشكلة حقيقية يواجهها مشرفو مجموعات القراءة. يحول التطبيق عملية تتبع القراءة اليدوية والمملة إلى تجربة تفاعلية ومحفزة، مع توفير أدوات تحليلية قوية للمشرفين لاتخاذ قرارات مبنية على البيانات.
      </p>
    </section>

    <section style="margin-bottom: 3rem; text-align: right;">
      <h2 style="font-size: 2rem; font-weight: bold; margin-bottom: 1rem; border-bottom: 2px solid hsl(var(--primary)); padding-bottom: 0.5rem; display: inline-block;">المشكلة</h2>
      <p style="line-height: 1.8; font-size: 1.1rem; color: hsl(var(--muted-foreground));">
        تواجه مجموعات القراءة التقليدية تحديات كبيرة تعيق نموها واستمراريتها، مثل العبء الإداري، انخفاض التفاعل، وغياب الرؤى التحليلية.
      </p>
    </section>
    
    <section style="margin-bottom: 3rem; text-align: right;">
      <h2 style="font-size: 2rem; font-weight: bold; margin-bottom: 1rem; border-bottom: 2px solid hsl(var(--primary)); padding-bottom: 0.5rem; display: inline-block;">الحل المقترح</h2>
      <p style="line-height: 1.8; font-size: 1.1rem; color: hsl(var(--muted-foreground)); margin-bottom: 1.5rem;">
        قمت ببناء تطبيق ويب سحابي يعالج كل هذه التحديات من خلال بنية تحتية ذكية ومؤتمتة بالكامل.
      </p>
      <img src="/images/projects/reading-marathon-app/architecture.png" alt="الهيكل المعماري للتطبيق" style="width: 100%; border-radius: var(--radius); border: 1px solid hsl(var(--border)); margin-top: 1rem;">
    </section>

    <section style="margin-bottom: 3rem;">
      <h2 style="font-size: 2rem; font-weight: bold; margin-bottom: 1rem; text-align: right; border-bottom: 2px solid hsl(var(--primary)); padding-bottom: 0.5rem; display: inline-block;">معرض صور التطبيق</h2>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
        <div style="border: 1px solid hsl(var(--border)); border-radius: var(--radius); overflow: hidden; transition: box-shadow 0.3s, transform 0.3s;" onmouseover="this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'; this.style.transform='translateY(-4px)';" onmouseout="this.style.boxShadow='none'; this.style.transform='none';">
          <img src="/images/projects/reading-marathon-app/dashboard.png" alt="لوحة التحكم العامة" style="width: 100%; display: block;">
          <p style="padding: 1rem; text-align: center; font-weight: 600; background-color: hsl(var(--card)); margin: 0;">لوحة التحكم العامة</p>
        </div>
        <div style="border: 1px solid hsl(var(--border)); border-radius: var(--radius); overflow: hidden; transition: box-shadow 0.3s, transform 0.3s;" onmouseover="this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'; this.style.transform='translateY(-4px)';" onmouseout="this.style.boxShadow='none'; this.style.transform='none';">
          <img src="/images/projects/reading-marathon-app/challenge-analytics.png" alt="تحليلات التحدي" style="width: 100%; display: block;">
          <p style="padding: 1rem; text-align: center; font-weight: 600; background-color: hsl(var(--card)); margin: 0;">تحليلات التحدي</p>
        </div>
        <div style="border: 1px solid hsl(var(--border)); border-radius: var(--radius); overflow: hidden; transition: box-shadow 0.3s, transform 0.3s;" onmouseover="this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'; this.style.transform='translateY(-4px)';" onmouseout="this.style.boxShadow='none'; this.style.transform='none';">
          <img src="/images/projects/reading-marathon-app/reader-card.png" alt="بطاقة أداء القارئ" style="width: 100%; display: block;">
          <p style="padding: 1rem; text-align: center; font-weight: 600; background-color: hsl(var(--card)); margin: 0;">بطاقة أداء القارئ</p>
        </div>
        <div style="border: 1px solid hsl(var(--border)); border-radius: var(--radius); overflow: hidden; transition: box-shadow 0.3s, transform 0.3s;" onmouseover="this.style.boxShadow='0 8px 25px rgba(0,0,0,0.1)'; this.style.transform='translateY(-4px)';" onmouseout="this.style.boxShadow='none'; this.style.transform='none';">
          <img src="/images/projects/reading-marathon-app/admin-panel.png" alt="لوحة الإدارة والإعدادات" style="width: 100%; display: block;">
          <p style="padding: 1rem; text-align: center; font-weight: 600; background-color: hsl(var(--card)); margin: 0;">لوحة الإدارة والإعدادات</p>
        </div>
      </div>
    </section>

    <section style="text-align: right;">
      <h2 style="font-size: 2rem; font-weight: bold; margin-bottom: 1rem; border-bottom: 2px solid hsl(var(--primary)); padding-bottom: 0.5rem; display: inline-block;">الدروس المستفادة</h2>
      <p style="line-height: 1.8; font-size: 1.1rem; color: hsl(var(--muted-foreground));">
        كان هذا المشروع رحلة تعليمية رائعة في بناء منتج برمجي متكامل. لقد أتاح لي الفرصة لتطبيق مهاراتي في **تحليل البيانات** في سياق أكبر يتطلب **هندسة برمجيات** و **تصميم أنظمة**. تعلمت أهمية التفكير في تجربة المستخدم (UX)، وأمان البيانات، وقابلية التوسع منذ اليوم الأول. يثبت هذا المشروع قدرتي على ترجمة فكرة إلى منتج حقيقي قابل للاستخدام، والتعامل مع تقنيات متعددة ومتكاملة لبناء حل قوي ومؤثر.
      </p>
    </section>
  </article>

</div>