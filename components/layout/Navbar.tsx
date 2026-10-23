"use client"

import { useState, useEffect } from "react"
import type { Route } from "next"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X, FileDown } from "lucide-react"
import ThemeToggle from "@/components/ThemeToggle"
import { siteConfig } from "@/config/site"
import { cn } from "@/lib/utils"

function isActivePath(pathname: string, path: string) {
  return path === "/" ? pathname === "/" : pathname === path || pathname.startsWith(path + "/")
}

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    onScroll()
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  useEffect(() => {
    setIsOpen(false)
  }, [pathname])

  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full border-b transition-colors duration-200",
        scrolled || isOpen
          ? "border-border bg-background/90 backdrop-blur-md"
          : "border-transparent bg-transparent",
      )}
    >
      <div className="mx-auto flex h-16 w-full max-w-[1240px] items-center justify-between px-5 md:px-8">
        <Link href="/" className="group flex items-center gap-3" aria-label="Ahmad Nayfeh, home">
          <span
            aria-hidden
            className="flex h-9 w-9 items-center justify-center border-[1.5px] border-foreground font-mono text-[13px] font-semibold tracking-tight transition-colors group-hover:border-accent group-hover:text-accent"
          >
            AN
          </span>
          <span className="flex flex-col leading-none">
            <span className="text-[15px] font-semibold tracking-tight">Ahmad Nayfeh</span>
            <span className="label mt-1 text-[9.5px]">Design Engineer</span>
          </span>
        </Link>

        <nav className="hidden items-center gap-1 md:flex" aria-label="Main">
          {siteConfig.nav.map((item) => {
            const active = isActivePath(pathname, item.path)
            return (
              <Link
                key={item.path}
                href={item.path as Route}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "group relative flex items-baseline gap-1.5 px-3 py-2 text-[13.5px] font-medium transition-colors",
                  active ? "text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                <span className={cn("font-mono text-[10px]", active ? "text-accent" : "text-muted-foreground/70")}>
                  {item.sheet}
                </span>
                {item.name}
                <span
                  aria-hidden
                  className={cn(
                    "absolute inset-x-3 -bottom-[13px] h-[2px] bg-accent transition-transform duration-300",
                    active ? "scale-x-100" : "scale-x-0 group-hover:scale-x-100",
                  )}
                  style={{ transformOrigin: "left" }}
                />
              </Link>
            )
          })}
          <span aria-hidden className="mx-3 h-5 w-px bg-border" />
          <a
            href={siteConfig.resume}
            target="_blank"
            rel="noopener noreferrer"
            className="mr-2 inline-flex items-center gap-1.5 font-mono text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground transition-colors hover:text-accent"
          >
            <FileDown size={13} />
            CV
          </a>
          <ThemeToggle />
        </nav>

        <div className="flex items-center gap-2 md:hidden">
          <ThemeToggle />
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center border border-border"
            onClick={() => setIsOpen((o) => !o)}
            aria-expanded={isOpen}
            aria-controls="mobile-nav"
            aria-label={isOpen ? "Close menu" : "Open menu"}
          >
            {isOpen ? <X size={17} /> : <Menu size={17} />}
          </button>
        </div>
      </div>

      {isOpen && (
        <nav id="mobile-nav" className="border-t border-border md:hidden" aria-label="Main">
          <ul className="mx-auto w-full max-w-[1240px] px-5 py-3">
            {siteConfig.nav.map((item) => {
              const active = isActivePath(pathname, item.path)
              return (
                <li key={item.path} className="border-b border-border last:border-b-0">
                  <Link
                    href={item.path as Route}
                    aria-current={active ? "page" : undefined}
                    className="flex items-baseline gap-3 py-3.5 text-base font-medium"
                  >
                    <span className={cn("font-mono text-[11px]", active ? "text-accent" : "text-muted-foreground")}>
                      {item.sheet}
                    </span>
                    <span className={active ? "text-foreground" : "text-muted-foreground"}>{item.name}</span>
                  </Link>
                </li>
              )
            })}
            <li>
              <a
                href={siteConfig.resume}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 py-3.5 text-base font-medium text-muted-foreground"
              >
                <FileDown size={15} className="text-accent" />
                Download CV
              </a>
            </li>
          </ul>
        </nav>
      )}
    </header>
  )
}
