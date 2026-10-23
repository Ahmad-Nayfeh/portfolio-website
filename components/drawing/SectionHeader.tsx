import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

interface SectionHeaderProps {
  number: string
  kicker: string
  title: ReactNode
  description?: ReactNode
  action?: ReactNode
  className?: string
  id?: string
}

/**
 * Section heading styled like a drawing zone marker: a numbered tag on a rule,
 * then the title and an optional description / action on the right.
 */
export default function SectionHeader({ number, kicker, title, description, action, className, id }: SectionHeaderProps) {
  return (
    <header className={cn("mb-10 md:mb-14", className)} id={id}>
      <div className="flex items-center gap-3">
        <span className="flex h-6 min-w-[2.25rem] items-center justify-center border-[1.5px] border-foreground px-1.5 font-mono text-[11px] font-semibold">
          {number}
        </span>
        <span className="label">{kicker}</span>
        <span aria-hidden className="h-px flex-1 bg-border" />
      </div>
      <div className="mt-6 grid grid-cols-1 items-end gap-x-10 gap-y-4 lg:grid-cols-12">
        <h2 className="font-display text-[clamp(1.75rem,3.6vw,2.75rem)] leading-[1.08] text-balance lg:col-span-7">
          {title}
        </h2>
        {(description || action) && (
          <div className="flex flex-col gap-4 lg:col-span-5 lg:items-end lg:text-right">
            {description && <p className="max-w-md text-[15px] leading-relaxed text-muted-foreground">{description}</p>}
            {action}
          </div>
        )}
      </div>
    </header>
  )
}
