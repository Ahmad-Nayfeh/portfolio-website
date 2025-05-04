"use client"

import type React from "react"

import { createContext, useContext, useEffect, useState } from "react"

type Theme = "dark" | "light" | "system"

type ThemeProviderProps = {
  children: React.ReactNode
  defaultTheme?: Theme
  storageKey?: string
  attribute?: string
  enableSystem?: boolean
  disableTransitionOnChange?: boolean
}

type ThemeProviderState = {
  theme: Theme
  setTheme: (theme: Theme) => void
}

const initialState: ThemeProviderState = {
  theme: "system",
  setTheme: () => null,
}

const ThemeProviderContext = createContext<ThemeProviderState>(initialState)

export function ThemeProvider({
  children,
  defaultTheme = "system",
  storageKey = "theme",
  attribute = "data-theme",
  enableSystem = true,
  disableTransitionOnChange = false,
  ...props
}: ThemeProviderProps) {
  const [theme, setTheme] = useState<Theme>(defaultTheme)

  useEffect(() => {
    const root = window.document.documentElement
    const savedTheme = localStorage.getItem(storageKey)

    if (savedTheme) {
      setTheme(savedTheme as Theme)
    } else if (enableSystem) {
      setTheme("system")
    }
  }, [enableSystem, storageKey])

  useEffect(() => {
    const root = window.document.documentElement

    if (disableTransitionOnChange) {
      root.classList.add("disable-transitions")
      window.setTimeout(() => {
        root.classList.remove("disable-transitions")
      }, 0)
    }

    if (theme === "system") {
      const systemTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"

      root.classList.toggle("dark", systemTheme === "dark")
      return
    }

    root.classList.toggle("dark", theme === "dark")
  }, [theme, disableTransitionOnChange])

  const value = {
    theme,
    setTheme: (theme: Theme) => {
      localStorage.setItem(storageKey, theme)
      setTheme(theme)
    },
  }

  return (
    <ThemeProviderContext.Provider {...props} value={value}>
      {children}
    </ThemeProviderContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeProviderContext)

  if (context === undefined) throw new Error("useTheme must be used within a ThemeProvider")

  return context
}
