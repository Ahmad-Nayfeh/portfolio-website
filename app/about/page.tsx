import type { Metadata } from "next"
import Image from "next/image"
import { FileDown, BookOpen } from "lucide-react"
import { getAboutContent } from "@/lib/content"
import Skills from "@/components/Skills"
import Experience from "@/components/Experience"
import Education from "@/components/Education"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import FadeIn from "@/components/ui/FadeIn"

export const metadata: Metadata = {
  title: "About | Ahmad Nayfeh",
  description: "Design engineer at Alfanar's RMU factory in Saudi Arabia.",
}

/** Books + papers currently on the reading stack. Hand-curated. */
const READING = [
  {
    title: "Probability Theory: The Logic of Science",
    author: "E. T. Jaynes",
    note: "The Bayesian foundation I keep returning to when a model misbehaves.",
  },
  {
    title: "The Art of Doing Science and Engineering",
    author: "Richard Hamming",
    note: "Hamming on how to think about problems worth solving.",
  },
  {
    title: "Designing Data-Intensive Applications",
    author: "Martin Kleppmann",
    note: "Still the clearest map of distributed systems thinking I've found.",
  },
  {
    title: "Deep Learning (Goodfellow et al.)",
    author: "Goodfellow, Bengio, Courville",
    note: "The theoretical spine — I reach for it when I need to understand why, not just how.",
  },
] as const

export default async function AboutPage() {
  const aboutContent = await getAboutContent()

  return (
    <div className="mx-auto w-full max-w-[1400px] px-6 pb-24 pt-12 md:px-10 lg:px-16">
      {/* Editorial masthead */}
      <FadeIn>
        <header className="mb-14 border-b border-border pb-10">
          <div className="mb-5 flex items-center gap-3">
            <span aria-hidden className="h-px w-8 bg-accent" />
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              Profile · About the engineer
            </span>
          </div>
          <h1 className="font-display text-display-xl text-balance">
            Hello, I&apos;m {aboutContent.name}
          </h1>
          {aboutContent.tagline && (
            <p className="font-display mt-6 max-w-2xl text-xl italic leading-snug text-muted-foreground md:text-2xl">
              {aboutContent.tagline}
            </p>
          )}
        </header>
      </FadeIn>

      {/* Bio + portrait spread */}
      <FadeIn delay={80}>
        <section className="grid grid-cols-12 gap-x-8 gap-y-10">
          <div className="col-span-12 lg:col-span-4">
            <div className="relative aspect-[4/5] overflow-hidden bg-secondary">
              <Image
                src="/images/profile.jpg"
                alt="Portrait of Ahmad Nayfeh"
                fill
                className="object-cover grayscale-[0.1]"
                priority
                sizes="(max-width: 1024px) 100vw, 33vw"
              />
            </div>
            <a
              href="/resume.pdf"
              target="_blank"
              rel="noreferrer"
              className="group mt-5 inline-flex w-full items-center justify-center gap-2 bg-foreground px-5 py-3 font-mono text-[11px] uppercase tracking-[0.2em] text-background transition-colors hover:bg-accent"
            >
              <FileDown size={14} />
              Download resume
            </a>
          </div>

          <div className="col-span-12 lg:col-span-8">
            <div className="prose prose-lg dark:prose-invert max-w-none">
              <MarkdownRenderer content={aboutContent.bio} />
            </div>
          </div>
        </section>
      </FadeIn>

      <FadeIn delay={40}>
        <Skills skills={aboutContent.skills} />
      </FadeIn>

      <FadeIn delay={40}>
        <Experience experience={aboutContent.experience} />
      </FadeIn>

      <FadeIn delay={40}>
        <Education education={aboutContent.education} />
      </FadeIn>

      {/* What I'm reading */}
      <FadeIn delay={40}>
        <section className="mt-24 border-t border-border pt-16">
          <div className="mb-10 flex items-center gap-3">
            <BookOpen size={16} className="text-accent" />
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              What I&apos;m reading
            </span>
          </div>
          <div className="grid grid-cols-1 gap-x-8 gap-y-0 md:grid-cols-2">
            {READING.map(({ title, author, note }, i) => (
              <div
                key={title}
                className="flex gap-4 border-b border-border py-6"
              >
                <span className="mt-0.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground/60">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <div>
                  <p className="font-display text-base leading-snug text-foreground">
                    {title}
                  </p>
                  <p className="mt-0.5 font-mono text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                    {author}
                  </p>
                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                    {note}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </section>
      </FadeIn>
    </div>
  )
}
