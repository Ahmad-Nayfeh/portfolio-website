"use client"

import { useState, useEffect } from "react"
import type { Route } from 'next';
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Menu, X } from "lucide-react"
import ThemeToggle from "@/components/ThemeToggle"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"


const navItems = [
  { name: "Home", path: "/" },
  { name: "Projects", path: "/projects" },
  { name: "Blog", path: "/blog" },
  { name: "About", path: "/about" },
]

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const pathname = usePathname()

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 10)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  useEffect(() => {
    // Close mobile menu when route changes
    setIsOpen(false)
  }, [pathname])

  return (
    <header
      className={cn(
        "sticky top-0 z-40 w-full transition-all duration-200",
        scrolled ? "bg-background/80 backdrop-blur-sm border-b" : "bg-transparent",
      )}
    >
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link href="/" className="text-xl font-bold">
            Ahmad Nayfeh
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-6">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path as Route}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-foreground/80",
                  pathname === item.path ? "text-foreground" : "text-foreground/60",
                )}
              >
                {item.name}
              </Link>
            ))}
            <ThemeToggle />
          </nav>

          {/* Mobile Navigation Toggle */}
          <div className="flex items-center gap-2 md:hidden">
            <ThemeToggle />
            <Button variant="ghost" size="icon" aria-label="Toggle Menu" onClick={() => setIsOpen(!isOpen)}>
              {isOpen ? <X size={20} /> : <Menu size={20} />}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {isOpen && (
        <div className="md:hidden border-t">
          <nav className="container mx-auto px-4 py-4 flex flex-col gap-4">
            {navItems.map((item) => (
              <Link
                key={item.path}
                href={item.path as Route}
                className={cn(
                  "py-2 text-sm font-medium transition-colors hover:text-foreground/80",
                  pathname === item.path ? "text-foreground" : "text-foreground/60",
                )}
              >
                {item.name}
              </Link>
            ))}
          </nav>
        </div>
      )}
    </header>
  )
}
