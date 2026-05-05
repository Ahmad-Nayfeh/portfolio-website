import Link from "next/link"
import type { Route } from "next"
import { Github, Linkedin, Mail, ArrowUp } from "lucide-react"

/**
 * Editorial footer / colophon.
 *
 * Sits as a quiet last spread: a small "colophon" header on the left, then a
 * three-column index, then a thin row with the copyright and a back-to-top
 * link. Nothing rounded, nothing shadowed — just rules and type.
 */
export default function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="mt-24 border-t border-border">
      <div className="mx-auto w-full max-w-[1400px] px-6 py-16 md:px-10 lg:px-16">
        <div className="grid grid-cols-12 gap-x-6 gap-y-12">
          {/* Wordmark + tagline */}
          <div className="col-span-12 md:col-span-5">
            <span className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              Colophon
            </span>
            <Link
              href="/"
              className="mt-3 block font-display text-3xl tracking-editorial text-foreground"
            >
              Ahmad Nayfeh
            </Link>
            <p className="mt-4 max-w-sm text-base text-muted-foreground">
              Design engineer at Alfanar&apos;s RMU factory in Saudi Arabia.
              Writing about engineering, AI, and the small surprises of
              building real systems.
            </p>
            <div className="mt-6 flex items-center gap-5">
              <Link
                href="https://github.com/Ahmad-Nayfeh"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub"
                className="text-muted-foreground transition-colors hover:text-accent"
              >
                <Github className="h-4 w-4" />
              </Link>
              <Link
                href="https://www.linkedin.com/in/ahmad-nayfeh2000/"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="LinkedIn"
                className="text-muted-foreground transition-colors hover:text-accent"
              >
                <Linkedin className="h-4 w-4" />
              </Link>
              <Link
                href="mailto:ahmadnayfeh2000@gmail.com"
                aria-label="Email"
                className="text-muted-foreground transition-colors hover:text-accent"
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
            <ul className="space-y-2 font-display text-base">
              {[
                { name: "Home", href: "/" },
                { name: "Projects", href: "/projects" },
                { name: "Notebook", href: "/blog" },
                { name: "About", href: "/about" },
              ].map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href as Route}
                    className="border-b border-transparent text-foreground transition-colors hover:border-accent hover:text-accent"
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
                className="block font-display text-base text-foreground transition-colors hover:text-accent"
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
              <span className="border-b border-foreground/30 transition-colors hover:border-accent">
                Download resume
              </span>
            </a>
          </div>
        </div>

        {/* Bottom rule row */}
        <div className="mt-16 flex flex-wrap items-center justify-between gap-3 border-t border-border pt-6 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          <p>
            &copy; {currentYear} Ahmad Nayfeh · All rights reserved
          </p>
          <a
            href="#top"
            className="group inline-flex items-center gap-1 text-foreground"
            aria-label="Back to top"
          >
            <span className="border-b border-border transition-colors group-hover:border-accent">
              Back to top
            </span>
            <ArrowUp size={11} className="transition-transform duration-300 group-hover:-translate-y-0.5" />
          </a>
        </div>
      </div>
    </footer>
  )
}
