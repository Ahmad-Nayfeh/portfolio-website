import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export interface TitleBlockCell {
  label: string
  value: ReactNode
  /** Column span on md+ screens (out of the grid's `cols`). */
  span?: number
  emphasis?: boolean
}

interface TitleBlockProps {
  cells: TitleBlockCell[]
  cols?: 2 | 3 | 4 | 6
  className?: string
}

const colClass: Record<number, string> = {
  2: "md:grid-cols-2",
  3: "md:grid-cols-3",
  4: "md:grid-cols-4",
  6: "md:grid-cols-6",
}

const spanClass: Record<number, string> = {
  1: "md:col-span-1",
  2: "md:col-span-2",
  3: "md:col-span-3",
  4: "md:col-span-4",
  5: "md:col-span-5",
  6: "md:col-span-6",
}

/**
 * A drawing title block: a ruled grid of labelled fields, like the box in the
 * corner of an engineering drawing. Outer border is heavier than the inner rules.
 */
export default function TitleBlock({ cells, cols = 4, className }: TitleBlockProps) {
  return (
    <div className={cn("border-[1.5px] border-foreground bg-card", className)}>
      <dl className={cn("grid grid-cols-2 [&>div]:border-b [&>div]:border-r [&>div]:border-border", colClass[cols])}>
        {cells.map((cell) => (
          <div
            key={cell.label}
            className={cn("col-span-2 min-w-0 px-3.5 py-2.5 sm:col-span-1", cell.span ? spanClass[cell.span] : undefined)}
          >
            <dt className="label text-[9.5px]">{cell.label}</dt>
            <dd
              className={cn(
                "mt-1 break-words text-[13.5px] leading-snug",
                cell.emphasis ? "font-semibold text-foreground" : "text-foreground/90",
              )}
            >
              {cell.value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
