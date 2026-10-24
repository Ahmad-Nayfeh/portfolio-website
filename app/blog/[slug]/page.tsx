import type { Metadata, Route } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ArrowLeft } from "lucide-react"
import { getPostBySlug, getRelatedPosts, getAllPosts } from "@/lib/content"
import RelatedPosts from "@/components/RelatedPosts"
import TableOfContents from "@/components/TableOfContents"
import { formatDate, formatTime, hasTimeComponent, extractHeadings } from "@/lib/utils"
import MdxRenderer from "@/components/MdxRenderer"

interface PageProps {
  params: Promise<{ slug: string }>
}

export async function generateStaticParams() {
  const posts = await getAllPosts()
  return posts.filter((post) => typeof post.slug === "string").map((post) => ({ slug: post.slug }))
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params
  const post = await getPostBySlug(slug)
  if (!post) return { title: "Entry not found" }
  return {
    title: post.title,
    description: post.excerpt,
    openGraph: { images: post.coverImage ? [post.coverImage] : [] },
  }
}

/** Hand-written posts start with their own "# Title"; the page header already shows it. */
function stripLeadingH1(md: string) {
  return md.replace(/^\s*#\s+[^\n]+\n/, "")
}

export default async function BlogPostPage({ params }: PageProps) {
  const { slug } = await params
  const post = await getPostBySlug(slug)
  if (!post) notFound()

  const relatedPosts = await getRelatedPosts(post, 3)
  const tocHeadings = post.content ? extractHeadings(post.content) : []
  const hasCover = post.coverImage && !post.coverImage.startsWith("/placeholder")

  return (
    <div className="mx-auto w-full max-w-[1240px] px-5 pb-16 pt-8 md:px-8 md:pt-12">
      <Link
        href={"/blog" as Route}
        className="group inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.16em] text-muted-foreground transition-colors hover:text-foreground"
      >
        <ArrowLeft size={12} className="transition-transform group-hover:-translate-x-0.5" />
        Notebook
      </Link>

      <article>
        <header className="mx-auto mt-8 max-w-[860px] border-b-[1.5px] border-foreground pb-10">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
            <time dateTime={post.date} className="label">
              {formatDate(post.date)}
              {hasTimeComponent(post.date) && ` · ${formatTime(post.date)}`}
            </time>
            {post.readTime && (
              <>
                <span aria-hidden className="text-border">|</span>
                <span className="label">{post.readTime}</span>
              </>
            )}
          </div>
          <h1 className="mt-5 font-display text-[clamp(2rem,4.8vw,3.4rem)] leading-[1.06] text-balance">{post.title}</h1>
          {post.excerpt && (
            <p className="mt-5 font-serif text-[1.2rem] italic leading-relaxed text-muted-foreground md:text-[1.3rem]">{post.excerpt}</p>
          )}
          {post.tags.length > 0 && (
            <div className="mt-6 flex flex-wrap gap-1.5">
              {post.tags.map((tag) => (
                <Link key={tag} href={`/blog?tag=${encodeURIComponent(tag)}` as Route} className="chip transition-colors hover:border-foreground hover:text-foreground">
                  {tag}
                </Link>
              ))}
            </div>
          )}
        </header>

        {hasCover && (
          <div className="sheet relative mx-auto mt-10 aspect-[16/9] max-w-[1080px] overflow-hidden">
            <Image src={post.coverImage} alt="" fill className="bg-card object-contain p-2" priority sizes="(max-width: 1080px) 100vw, 1080px" />
          </div>
        )}

        <div className="mx-auto mt-12 grid max-w-[1080px] grid-cols-1 gap-x-12 xl:grid-cols-[minmax(0,760px)_240px] xl:justify-center">
          <div className="article-body min-w-0">{post.content && <MdxRenderer source={stripLeadingH1(post.content)} />}</div>
          <div className="hidden xl:block">
            <TableOfContents headings={tocHeadings} />
          </div>
        </div>

        <div className="mx-auto mt-16 max-w-[860px] border-t border-border pt-6">
          <p className="label">End of entry · Ahmad Nayfeh · {formatDate(post.date)}</p>
        </div>
      </article>

      {relatedPosts.length > 0 && (
        <section className="mx-auto mt-20 max-w-[1080px]">
          <div className="mb-6 flex items-center gap-3">
            <span className="label">Keep reading</span>
            <span aria-hidden className="h-px flex-1 bg-border" />
          </div>
          <RelatedPosts posts={relatedPosts} />
        </section>
      )}
    </div>
  )
}
