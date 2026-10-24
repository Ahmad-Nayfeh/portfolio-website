import type { Route } from "next"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"

export default function NotFound() {
  return (
    <div className="mx-auto flex w-full max-w-[1240px] flex-col items-start px-5 py-24 md:px-8 md:py-32">
      <span className="label">Error 404</span>
      <h1 className="mt-4 font-display text-[clamp(2.5rem,6vw,4.25rem)] leading-[1]">Page not found.</h1>
      <p className="mt-5 max-w-lg text-[1.1rem] leading-relaxed text-muted-foreground">
        This page doesn&apos;t exist. It may have been moved or removed.
      </p>
      <Link href={"/" as Route} className="btn btn-primary mt-8">
        <ArrowLeft size={14} /> Back to home
      </Link>
    </div>
  )
}
