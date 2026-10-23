import type { NowItem } from "@/types"

/** "Operating state" strip: what I'm doing right now, as indicator lamps. */
export default function NowPanel({ items }: { items: NowItem[] }) {
  if (items.length === 0) return null
  return (
    <section aria-label="Now" className="mx-auto w-full max-w-[1240px] px-5 md:px-8">
      <div className="border-y-[1.5px] border-foreground bg-card/70">
        <div className="grid grid-cols-1 md:grid-cols-[auto_1fr_1fr_1fr]">
          <div className="flex items-center gap-2 border-b border-border px-4 py-3 md:border-b-0 md:border-r md:px-5">
            <span className="label text-foreground">Now</span>
          </div>
          {items.map((item) => (
            <div
              key={item.label}
              className="flex items-start gap-3 border-b border-border px-4 py-4 last:border-b-0 md:border-b-0 md:border-r md:px-5 md:last:border-r-0"
            >
              <span className="lamp mt-[5px] shrink-0" aria-hidden />
              <div>
                <div className="label text-[10px]">{item.label}</div>
                <p className="mt-1 text-[14px] leading-snug text-foreground/90">{item.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
