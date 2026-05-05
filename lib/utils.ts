import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/**
 * Format a date string for display.
 *
 * Accepts both date-only ("2026-05-02") and ISO 8601 with offset
 * ("2026-05-02T13:00:00+03:00"). Renders just the date — month, day, year.
 * For the date + time combination see `formatDateTime`.
 *
 * Hand-written posts use plain "YYYY-MM-DD"; auto-generated pipeline posts
 * write a full ISO timestamp so we can display the publish *time* on the
 * post page.
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(date)
}

/**
 * Returns true when the input string includes a time component (i.e. ISO
 * 8601 with the "T" separator). Used by post pages to decide whether to
 * render the time alongside the date.
 */
export function hasTimeComponent(dateString: string): boolean {
  return /T\d{2}:\d{2}/.test(dateString)
}

/**
 * Format a date with its time-of-day component, in Asia/Riyadh time.
 *
 * Returns the time only (e.g., "10:00 AST"). Pair with `formatDate` to get
 * the full "May 3, 2026 · 10:00 AST" string. We render in Riyadh time
 * because that's where the pipeline runs are anchored — readers should see
 * the publish moment in the author's local time, not their own.
 */
export function formatTime(dateString: string): string {
  if (!hasTimeComponent(dateString)) return ""
  const date = new Date(dateString)
  // Asia/Riyadh is UTC+3, no DST. Intl handles the conversion regardless of
  // the runtime's local timezone (Vercel's edge can be anywhere).
  return new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: "Asia/Riyadh",
    timeZoneName: "short",
  }).format(date)
}

/**
 * Convenience for the common "date · time" rendering on post pages. If the
 * input has no time component, falls through to just the date.
 */
export function formatDateTime(dateString: string): string {
  const datePart = formatDate(dateString)
  const timePart = formatTime(dateString)
  return timePart ? `${datePart} · ${timePart}` : datePart
}
