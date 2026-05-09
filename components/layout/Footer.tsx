import Link from "next/link"
import type { Route } from "next"
import { Github, Linkedin, Mail, ArrowUp } from "lucide-react"

export default function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="mt-24 border-t" style={{ borderColor: "hsl(var(--border))" }}>
      <div className="mx-auto w-full max-w-[1400px] px-6 py-16 md:px-10 lg:px-16">
        <div className="grid grid-cols-12 gap-x-6 gap-y-12">
          {/* Wordmark + tagline */}
          <div className="col-span-12 md:col-span-5">
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Colophon
            </span>
            <Link
              href="/"
              className="mt-3 block text-3xl font-bold tracking-tight text-foreground"
            >
              Ahmad Nayfeh
            </Link>
            <p className="mt-4 max-w-sm text-base text-muted-foreground">
              Designing and deploying end-to-end intelligent systems.
              Writing about engineering, AI, and the small surprises of
              building real systems.
            </p>
            <div className="mt-6 flex items-center gap-5">
              <Link
                href="https://github.com/Ahmad-Nayfeh"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub"
                className="transition-colors"
                style={{ color: "hsl(var(--muted-foreground))" }}
              >
                <Github className="h-4 w-4" />
              </Link>
              <Link
                href="https://www.linkedin.com/in/ahmad-nayfeh2000/"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="LinkedIn"
                className="transition-colors"
                style={{ color: "hsl(var(--muted-foreground))" }}
              >
                <Linkedin className="h-4 w-4" />
              </Link>
              <Link
                href="mailto:ahmadnayfeh2000@gmail.com"
                aria-label="Email"
                className="transition-colors"
                style={{ color: "hsl(var(--muted-foreground))" }}
              >
                <Mail className="h-4 w-4" />
              </Link>
            </div>
          </div>

          {/* Index column */}
          <div className="col-span-6 md:col-span-3">
            <span className="mb-4 block font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              The Index
            </span>
            <ul className="space-y-2 text-base font-bold tracking-tight">
              {[
                { name: "Home", href: "/" },
                { name: "Projects", href: "/projects" },
                { name: "Notebook", href: "/blog" },
                { name: "About", href: "/about" },
              ].map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href as Route}
                    className="border-b border-transparent text-foreground transition-colors"
                    style={{ borderColor: "transparent" }}
                  >
                    {item.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact column */}
          <div className="col-span-6 md:col-span-4">
            <span className="mb-4 block font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Get in touch
            </span>
            <address className="not-italic">
              <a
                href="mailto:ahmadnayfeh2000@gmail.com"
                className="block text-base font-bold tracking-tight text-foreground transition-colors"
              >
                ahmadnayfeh2000@gmail.com
              </a>
              <p className="mt-2 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                Riyadh · Saudi Arabia
              </p>
            </address>

            <a
              href="/resume.pdf"
              target="_blank"
              rel="noreferrer"
              className="mt-6 inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-foreground"
            >
              <span
                className="border-b transition-colors"
                style={{
                  borderColor: "hsl(var(--section-accent) / 0.35)",
                }}
              >
                Download resume
              </span>
            </a>
          </div>
        </div>

        {/* Bottom rule row */}
        <div className="mt-16 flex flex-wrap items-center justify-between gap-3 pt-6 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground"
          style={{ borderTop: "1px solid hsl(var(--border))" }}
        >
          <p>
            &copy; {currentYear} Ahmad Nayfeh · All rights reserved
          </p>
          <a
            href="#top"
            className="group inline-flex items-center gap-1 text-foreground"
            aria-label="Back to top"
          >
            <span
              className="border-b transition-colors group-hover:border-current"
              style={{
                borderColor: "hsl(var(--section-accent) / 0.35)",
              }}
            >
              Back to top
            </span>
            <ArrowUp size={11} className="transition-transform duration-300 group-hover:-translate-y-0.5" />
          </a>
        </div>
      </div>
    </footer>
  )
}
