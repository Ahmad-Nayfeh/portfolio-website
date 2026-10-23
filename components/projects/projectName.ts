import type { Project } from "@/types"

/**
 * Project titles are written as "Name: tagline". Split them so cards can show
 * the name large and the tagline small. Titles without a colon are returned
 * whole as the name.
 */
export function splitTitle(title: string): { name: string; tagline: string } {
  const clean = title.replace(/^[^\p{L}\p{N}]+/u, "").trim()
  const idx = clean.indexOf(":")
  if (idx > 0 && idx < 40) {
    return { name: clean.slice(0, idx).trim(), tagline: clean.slice(idx + 1).trim() }
  }
  return { name: clean, tagline: "" }
}

export function projectYear(p: Pick<Project, "date">): string {
  const d = new Date(p.date)
  return Number.isNaN(d.getTime()) ? "" : String(d.getFullYear())
}

export function hasLink(url?: string): url is string {
  return Boolean(url && url !== "#")
}
