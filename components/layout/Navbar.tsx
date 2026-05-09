"use client"

import { useState, useEffect } from "react"
import type { Route } from "next"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const navItems = [
  { name: "Index", path: "/" },
  { name: "Projects", path: "/projects" },
  { name: "Notebook", path: "/blog" },
  { name: "About", path: "/about" },
]

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8)
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

  useEffect(() => {
    setIsOpen(false)
  }, [pathname])

  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full transition-all duration-200",
        scrolled
          ? "glass-card-strong border-b border-border/50"
          : "bg-transparent",
      )}
    >
      <div className="mx-auto w-full max-w-[1400px] px-4 md:px-10 lg:px-16">
        <div className="flex h-16 items-center justify-between md:h-20">
          <Link
            href="/"
            className="font-display text-xl tracking-tight md:text-2xl glow-text"
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
                      "absolute -bottom-1.5 left-0 h-px transition-[width] duration-300 ease-out",
                      isActive ? "w-full" : "w-0 group-hover:w-full",
                    )}
                    style={{
                      backgroundColor: isActive ? "hsl(var(--section-accent))" : "hsl(var(--section-accent))",
                    }}
                  />
                </Link>
              )
            })}
          </nav>

          {/* Mobile Navigation Toggle */}
          <div className="flex items-center gap-1 md:hidden">
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
                  <span className="mr-3" style={{ color: "hsl(var(--section-accent))" }}>
                    {isActive ? "▸" : "·"}
                  </span>
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
