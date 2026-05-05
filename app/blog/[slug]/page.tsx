import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ArrowLeft, Clock } from "lucide-react"
import { getPostBySlug, getRelatedPosts, getAllPosts } from "@/lib/content"
import RelatedPosts from "@/components/RelatedPosts"
import { formatDate, formatTime, hasTimeComponent } from "@/lib/utils"
import MdxRenderer from "@/components/MdxRenderer"
import type { FullPost } from "@/types"

interface BlogPageParams {
  slug: string
}

export async function generateStaticParams(): Promise<Array<BlogPageParams>> {
  const posts = await getAllPosts()
  return posts
    .filter((post) => typeof post.slug === "string")
    .map((post) => ({ slug: post.slug }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params
  const post = await getPostBySlug(slug)
  if (!post) return { title: "Post Not Found" }
  return {
    title: `${post.title} | Ahmad Nayfeh`,
    description: post.excerpt,
  }
}

export default async function BlogPostPage({ params }: PageProps) {
  const { slug } = await params
  const post: FullPost | null = await getPostBySlug(slug)
  if (!post) notFound()

  const relatedPosts = await getRelatedPosts(post, 3)

  return (
    <div className="mx-auto w-full max-w-[1400px] px-6 pb-20 pt-8 md:px-10 lg:px-16">
      <Link
        href="/blog"
        className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.2em] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft size={12} className="transition-transform duration-300 group-hover:-translate-x-0.5" />
        <span className="border-b border-border transition-colors group-hover:border-accent">
          Back to the notebook
        </span>
      </Link>

      <article>
        {/* === Editorial header — sits above the article column,
              full-bleed feel without leaving the grid. === */}
        <header className="mt-12 grid grid-cols-12 gap-x-6 border-b border-border pb-10">
          {/* Meta column on the left */}
          <div className="col-span-12 lg:col-span-3">
            <div className="flex flex-row gap-x-6 lg:flex-col lg:gap-y-5">
              {post.date && (
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    Published
                  </div>
                  <time
                    dateTime={post.date}
                    className="mt-1 block font-display text-base text-foreground"
                  >
                    {formatDate(post.date)}
                  </time>
                  {hasTimeComponent(post.date) && (
                    <span className="mt-0.5 block font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                      {formatTime(post.date)}
                    </span>
                  )}
                </div>
              )}
              {post.readTime && (
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    Read
                  </div>
                  <span className="mt-1 inline-flex items-center gap-1.5 font-display text-base text-foreground">
                    <Clock size={13} className="text-muted-foreground" />
                    {post.readTime}
                  </span>
                </div>
              )}
              {post.tags && post.tags.length > 0 && (
                <div className="hidden lg:block">
                  <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                    Tagged
                  </div>
                  <div className="mt-2 flex flex-wrap gap-x-2 gap-y-1 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
                    {post.tags.map((tag) => (
                      <span key={tag} className="border-b border-border">
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Title column on the right */}
          <div className="col-span-12 mt-8 lg:col-span-9 lg:mt-0">
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-accent">
              The Notebook
            </span>
            <h1 className="font-display text-display-xl mt-4 text-balance">
              {post.title}
            </h1>
            {post.excerpt && (
              <p className="font-display mt-6 max-w-2xl text-xl italic leading-snug text-muted-foreground md:text-2xl">
                {post.excerpt}
              </p>
            )}
            {/* Tags on mobile — kept here so they appear after the title. */}
            {post.tags && post.tags.length > 0 && (
              <div className="mt-6 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground lg:hidden">
                {post.tags.map((tag) => (
                  <span key={tag} className="border-b border-border">
                    #{tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        </header>

        {/* Cover image full editorial width */}
        {post.coverImage && (
          <div className="relative my-10 aspect-[16/9] overflow-hidden bg-secondary md:my-14">
            <Image
              src={post.coverImage}
              alt={post.title}
              fill
              className="object-cover"
              priority
              sizes="(max-width: 1400px) 100vw, 1400px"
            />
          </div>
        )}

        {/* Body — single editorial column, generous measure. */}
        <div className="mx-auto max-w-2xl">
          {post.content && (
            <div className="prose prose-lg dark:prose-invert max-w-none editorial-body">
              <MdxRenderer source={post.content} />
            </div>
          )}
        </div>

        {/* Footer rule with a single mono signpost. */}
        <div className="mx-auto mt-20 max-w-2xl border-t border-border pt-8">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            End of entry · Ahmad Nayfeh · {post.date ? formatDate(post.date) : ""}
          </p>
        </div>
      </article>

      {relatedPosts.length > 0 && (
        <section className="mt-24 border-t border-border pt-16">
          <div className="mb-10 flex items-center gap-3">
            <span aria-hidden className="h-px w-8 bg-accent" />
            <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
              Keep reading
            </span>
          </div>
          <h2 className="font-display text-display-md mb-10 text-balance">
            Related entries from the notebook
          </h2>
          <RelatedPosts posts={relatedPosts} />
        </section>
      )}
    </div>
  )
}

type PageProps = {
  params: Promise<{ slug: string }>
  searchParams?: Promise<{ [key: string]: string | string[] | undefined }>
}
