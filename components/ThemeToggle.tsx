"use client"

import { useEffect, useState } from "react"
import { useTheme } from "next-themes"
import { Moon, Sun } from "lucide-react"

/** Paper (light) / blueprint (dark) switch. */
export default function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  useEffect(() => setMounted(true), [])

  const isDark = mounted && resolvedTheme === "dark"

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="inline-flex h-9 w-9 items-center justify-center border border-border text-muted-foreground transition-colors hover:border-foreground hover:text-foreground"
      aria-label={isDark ? "Switch to paper (light) theme" : "Switch to blueprint (dark) theme"}
      title={isDark ? "Paper" : "Blueprint"}
    >
      {mounted ? isDark ? <Sun size={15} /> : <Moon size={15} /> : <span className="h-[15px] w-[15px]" />}
    </button>
  )
}
