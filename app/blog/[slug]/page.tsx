import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ArrowLeft, Clock } from "lucide-react"
import { getPostBySlug, getRelatedPosts, getAllPosts } from "@/lib/content"
import { Badge } from "@/components/ui/badge"
import RelatedPosts from "@/components/RelatedPosts"
import { formatDate } from "@/lib/utils"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import type { FullPost } from "@/types"

// Define the parameter structure for generateStaticParams
interface BlogPageParams {
  slug: string;
}

// *** Define the Props type for Page and generateMetadata (using Promise for params) ***
type PageProps = {
  params: Promise<{ slug: string }>;
  searchParams?: Promise<{ [key: string]: string | string[] | undefined }>;
};

// generateStaticParams remains largely the same, returning slugs
export async function generateStaticParams(): Promise<Array<BlogPageParams>> {
  const posts = await getAllPosts()
  // Ensure slugs are valid strings
  return posts.filter(post => typeof post.slug === 'string').map((post) => ({
    slug: post.slug,
  }))
}

// *** Updated generateMetadata to use PageProps and await params ***
export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  // Await the params Promise
  const resolvedParams = await params;
  const { slug } = resolvedParams; // Get slug from resolved params

  const post = await getPostBySlug(slug)

  if (!post) {
    return {
      title: "Post Not Found",
    }
  }

  return {
    title: `${post.title} | Blog`,
    description: post.excerpt,
    // Consider adding Open Graph metadata here too
  }
}

// *** Updated Page component to use PageProps and await params ***
export default async function BlogPostPage({ params }: PageProps) {
  // Await the params Promise
  const resolvedParams = await params;
  const { slug } = resolvedParams; // Get slug from resolved params

  const post: FullPost | null = await getPostBySlug(slug)

  if (!post) {
    notFound()
  }

  // Fetch related posts AFTER getting the current post successfully
  const relatedPosts = await getRelatedPosts(post, 3)

  return (
    <div className="container mx-auto px-4 py-12">
      <Link
        href="/blog"
        className="mb-8 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft size={16} />
        <span>Back to Blog</span>
      </Link>

      <article className="max-w-3xl mx-auto">
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl mb-4">
          {post.title}
        </h1>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-8 text-sm text-muted-foreground">
          {/* Ensure date is valid before formatting */}
          {post.date && <time dateTime={post.date}>{formatDate(post.date)}</time>}
          {post.readTime && (
            <div className="flex items-center gap-1">
              <Clock size={16} />
              <span>{post.readTime}</span>
            </div>
          )}
        </div>

        {post.coverImage && (
            <div className="relative aspect-video mb-8 rounded-lg overflow-hidden border">
              <Image
                src={post.coverImage}
                alt={post.title}
                fill
                className="object-cover"
                priority
                sizes="(max-width: 768px) 100vw, (max-width: 1024px) 75vw, 800px"
              />
            </div>
        )}

        <div className="flex flex-wrap gap-2 mb-8">
          {post.tags?.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>

        {/* Ensure content exists before rendering */}
        {post.content && <MarkdownRenderer content={post.content} />}
      </article>

      {relatedPosts.length > 0 && (
        <section className="max-w-3xl mx-auto mt-16 pt-8 border-t">
          <h2 className="text-2xl font-bold mb-8">Related Posts</h2>
          <RelatedPosts posts={relatedPosts} />
        </section>
      )}
    </div>
  )
}