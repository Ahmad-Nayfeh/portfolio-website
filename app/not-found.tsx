import type { Route } from "next"
import Link from "next/link"
import { ArrowLeft } from "lucide-react"
import TitleBlock from "@/components/drawing/TitleBlock"

export default function NotFound() {
  return (
    <div className="mx-auto flex w-full max-w-[1240px] flex-col items-start px-5 py-24 md:px-8 md:py-32">
      <span className="label">Error 404</span>
      <h1 className="mt-4 font-display text-[clamp(2.5rem,6vw,4.25rem)] leading-[1]">Sheet not found.</h1>
      <p className="mt-5 max-w-lg text-[1.1rem] leading-relaxed text-muted-foreground">
        This page isn&apos;t in the drawing set. It may have been moved or removed in a later revision.
      </p>
      <TitleBlock
        className="mt-10 w-full max-w-xl"
        cols={2}
        cells={[
          { label: "Status", value: "Open circuit", emphasis: true },
          { label: "Action", value: "Return to the index" },
        ]}
      />
      <Link href={"/" as Route} className="btn btn-primary mt-8">
        <ArrowLeft size={14} /> Back to index
      </Link>
    </div>
  )
}
