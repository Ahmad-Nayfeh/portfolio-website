import type { Route } from "next"
import Link from "next/link"
import { ArrowUpRight } from "lucide-react"
import type { Post } from "@/types"

function shortDate(dateString: string) {
  const d = new Date(dateString)
  if (Number.isNaN(d.getTime())) return ""
  return new Intl.DateTimeFormat("en-US", { month: "short", year: "numeric" }).format(d)
}

/** One notebook entry, laid out as a row in a drawing register. */
export default function BlogCard({ post, index, headingLevel = 3 }: { post: Post; index?: number; headingLevel?: 2 | 3 }) {
  const Heading = headingLevel === 2 ? "h2" : "h3"
  return (
    <Link
      href={`/blog/${post.slug}` as Route}
      className="group grid grid-cols-[3.25rem_1fr] gap-x-4 gap-y-1 border-b border-border py-6 transition-colors hover:bg-card/70 md:grid-cols-[5.5rem_8rem_1fr_auto] md:items-baseline md:gap-x-6 md:px-3"
    >
      <span className="font-mono text-[11px] font-semibold text-accent md:pt-1">
        N-{String((index ?? 0) + 1).padStart(2, "0")}
      </span>
      <time dateTime={post.date} className="label col-start-2 row-start-1 self-center md:col-start-auto md:row-start-auto md:self-baseline md:pt-1">
        {shortDate(post.date)}
      </time>
      <div className="col-span-2 md:col-span-1">
        <Heading className="text-[1.15rem] font-semibold leading-snug tracking-tight md:text-[1.25rem]">
          <span className="redline">{post.title}</span>
        </Heading>
        {post.excerpt && (
          <p className="mt-2 line-clamp-2 max-w-3xl text-[15px] leading-relaxed text-muted-foreground">{post.excerpt}</p>
        )}
        {post.tags.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {post.tags.slice(0, 4).map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
        )}
      </div>
      <span className="hidden items-center gap-2 md:flex">
        <span className="label text-[10px]">{post.readTime}</span>
        <ArrowUpRight size={17} className="text-muted-foreground transition-colors group-hover:text-accent" />
      </span>
    </Link>
  )
}
