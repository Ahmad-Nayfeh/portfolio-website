import type React from "react"
import type { Metadata } from "next"
import { Inter, Newsreader, JetBrains_Mono } from "next/font/google"
import "./globals.css"
import "katex/dist/katex.min.css"
import Navbar from "@/components/layout/Navbar"
import Footer from "@/components/layout/Footer"
import { ThemeProvider } from "@/components/theme-provider"

// Body / UI sans — Inter, tuned for screens.
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-sans",
})

// Editorial display serif — Newsreader. Variable axes for weight + optical size,
// gives the magazine-style headline voice without feeling antique.
const newsreader = Newsreader({
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
  variable: "--font-serif",
})

// Monospace — JetBrains Mono. Slightly humanist; good ligatures for code blocks.
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600"],
  variable: "--font-mono",
})

export const metadata: Metadata = {
  title: "Ahmad Nayfeh — Design Engineer",
  description:
    "Notes on engineering, AI, and systems thinking from a design engineer at Alfanar's RMU factory.",
  generator: "Next.js",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${newsreader.variable} ${jetbrainsMono.variable}`}
    >
      <head>
        <meta name="google-site-verification" content="TRmiP4XM7rQHNMBz7LJs_ZTRzVb46pLD0LJlJ-hz8QU" />
      </head>

      <body className="font-sans antialiased">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <div className="flex min-h-screen flex-col">
            <Navbar />
            <main className="flex-1">{children}</main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  )
}