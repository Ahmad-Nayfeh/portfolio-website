import Image from "next/image"
import Link from "next/link"
import { Clock } from "lucide-react"
import type { Post } from "@/types"
import { formatDate } from "@/lib/utils"

interface BlogCardProps {
  post: Post
  index?: number
}

export default function BlogCard({ post, index }: BlogCardProps) {
  const tags = post.tags ?? []
  return (
    <article className="group flex flex-col">
      <Link
        href={`/blog/${post.slug}`}
        className="relative block aspect-[16/9] w-full overflow-hidden rounded-lg"
        style={{ backgroundColor: "hsl(var(--card))" }}
      >
        <Image
          src={post.coverImage || "/placeholder.svg"}
          alt={post.title}
          fill
          sizes="(max-width: 768px) 100vw, (max-width: 1024px) 50vw, 33vw"
          className="object-cover transition-all duration-700 ease-out group-hover:scale-[1.03]"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent" />
        {typeof index === "number" && (
          <span className="absolute left-3 top-3 rounded px-2 py-1 font-mono text-[10px] uppercase tracking-[0.2em] backdrop-blur-sm"
            style={{
              backgroundColor: "hsl(var(--background) / 0.6)",
              color: "hsl(var(--foreground))",
              border: "1px solid hsl(var(--border) / 0.5)",
            }}
          >
            № {String(index + 1).padStart(2, "0")}
          </span>
        )}
      </Link>

      <div className="flex flex-1 flex-col pt-5">
        <div className="mb-3 flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
          {post.date && <time dateTime={post.date}>{formatDate(post.date)}</time>}
          {post.date && post.readTime && <span aria-hidden>·</span>}
          {post.readTime && (
            <span className="inline-flex items-center gap-1">
              <Clock size={11} />
              {post.readTime}
            </span>
          )}
        </div>

        <Link href={`/blog/${post.slug}`} className="block">
          <h3 className="text-2xl font-bold leading-tight tracking-tight text-balance text-foreground transition-colors">
            {post.title}
          </h3>
        </Link>

        {post.excerpt && (
          <p className="mt-3 line-clamp-3 text-base leading-relaxed text-muted-foreground">
            {post.excerpt}
          </p>
        )}

        {tags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            {tags.slice(0, 3).map((tag) => (
              <span key={tag}>#{tag}</span>
            ))}
            {tags.length > 3 && <span>+{tags.length - 3}</span>}
          </div>
        )}
      </div>
    </article>
  )
}
