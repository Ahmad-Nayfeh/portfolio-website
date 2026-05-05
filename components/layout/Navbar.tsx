"use client"

/**
 * Editorial masthead navigation.
 *
 * The top of the site reads like a magazine masthead: a thin meta-row above
 * (date / edition), then the brand wordmark in display serif, with the nav
 * laid out as small caps. Active route is marked by a cobalt rule under the
 * label, not a color swap — keeps the bar quiet and the accent meaningful.
 */
import { useState, useEffect } from "react"
import type { Route } from "next"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X } from "lucide-react"
import ThemeToggle from "@/components/ThemeToggle"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const navItems = [
  { name: "Index", path: "/" },
  { name: "Projects", path: "/projects" },
  { name: "Notebook", path: "/blog" },
  { name: "About", path: "/about" },
]

// Tiny utility — formats today as "MAY 02, 2026" for the meta-row.
function formatToday(): string {
  return new Date()
    .toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" })
    .toUpperCase()
}

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const [today, setToday] = useState("")
  const pathname = usePathname()

  useEffect(() => {
    setToday(formatToday())
    const onScroll = () => setScrolled(window.scrollY > 8)
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  // Close mobile menu when route changes.
  useEffect(() => {
    setIsOpen(false)
  }, [pathname])

  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full transition-[background-color,backdrop-filter,border-color] duration-200",
        scrolled
          ? "bg-background/80 backdrop-blur-md border-b border-border/80"
          : "bg-background border-b border-transparent",
      )}
    >
      {/* Meta row — date, edition number, slogan. Hidden when scrolled to keep
          the masthead compact during reading. */}
      <div
        className={cn(
          "border-b border-border/60 transition-all overflow-hidden",
          scrolled ? "h-0 opacity-0" : "h-7 opacity-100",
        )}
      >
        <div className="container mx-auto flex h-7 items-center justify-between px-4 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          <span suppressHydrationWarning>{today || "—"}</span>
          <span className="hidden sm:inline">Engineering · AI · Systems</span>
          <span>Vol. 01</span>
        </div>
      </div>

      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between md:h-20">
          <Link
            href="/"
            className="font-display text-xl tracking-tightest md:text-2xl"
            aria-label="Ahmad Nayfeh — home"
          >
            Ahmad Nayfeh
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden items-center gap-8 md:flex">
            {navItems.map((item) => {
              const isActive =
                item.path === "/"
                  ? pathname === "/"
                  : pathname === item.path || pathname.startsWith(item.path + "/")
              return (
                <Link
                  key={item.path}
                  href={item.path as Route}
                  className={cn(
                    "group relative font-mono text-[11px] uppercase tracking-[0.18em] transition-colors",
                    isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {item.name}
                  <span
                    aria-hidden
                    className={cn(
                      "absolute -bottom-1.5 left-0 h-px bg-accent transition-[width] duration-300 ease-out",
                      isActive ? "w-full" : "w-0 group-hover:w-full",
                    )}
                  />
                </Link>
              )
            })}
            <span aria-hidden className="h-4 w-px bg-border" />
            <ThemeToggle />
          </nav>

          {/* Mobile Navigation Toggle */}
          <div className="flex items-center gap-1 md:hidden">
            <ThemeToggle />
            <Button
              variant="ghost"
              size="icon"
              aria-label="Toggle Menu"
              onClick={() => setIsOpen((v) => !v)}
            >
              {isOpen ? <X size={20} /> : <Menu size={20} />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isOpen && (
        <div className="border-t border-border/60 md:hidden">
          <nav className="container mx-auto flex flex-col px-4 py-4">
            {navItems.map((item) => {
              const isActive =
                item.path === "/"
                  ? pathname === "/"
                  : pathname === item.path || pathname.startsWith(item.path + "/")
              return (
                <Link
                  key={item.path}
                  href={item.path as Route}
                  className={cn(
                    "border-b border-border/40 py-3 font-mono text-xs uppercase tracking-[0.18em] last:border-b-0",
                    isActive ? "text-foreground" : "text-muted-foreground",
                  )}
                >
                  <span className="mr-3 text-accent">{isActive ? "▸" : "·"}</span>
                  {item.name}
                </Link>
              )
            })}
          </nav>
        </div>
      )}
    </header>
  )
}
