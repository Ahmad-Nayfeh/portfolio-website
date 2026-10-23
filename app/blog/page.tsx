import type { Metadata } from "next"
import { Suspense } from "react"
import BlogGrid from "@/components/BlogGrid"
import { getAllPosts } from "@/lib/content"

export const metadata: Metadata = {
  title: "Notebook",
  description: "Long-form explainers on signals, images, and how machines learn.",
}

function BlogGridSkeleton() {
  return (
    <div className="border-t-[1.5px] border-foreground">
      {[...Array(4)].map((_, i) => (
        <div key={i} className="flex flex-col gap-3 border-b border-border py-6">
          <div className="h-3 w-24 animate-pulse bg-secondary" />
          <div className="h-5 w-2/3 animate-pulse bg-secondary" />
          <div className="h-3 w-full animate-pulse bg-secondary" />
        </div>
      ))}
    </div>
  )
}

export default async function BlogPage() {
  const posts = await getAllPosts()

  return (
    <div className="mx-auto w-full max-w-[1240px] px-5 pb-16 pt-10 md:px-8 md:pt-16">
      <header className="mb-12 grid grid-cols-1 gap-6 lg:grid-cols-12 lg:items-end">
        <div className="lg:col-span-7">
          <span className="label">Sheet 03 · Notebook</span>
          <h1 className="mt-4 font-display text-[clamp(2.5rem,6vw,4.25rem)] leading-[1]">Notebook</h1>
        </div>
        <p className="max-w-md text-[1.05rem] leading-relaxed text-muted-foreground lg:col-span-5 lg:justify-self-end lg:text-right">
          Writing is how I check that I actually understand something. These are long-form explainers on signals, images, and how machines learn.
        </p>
      </header>

      <Suspense fallback={<BlogGridSkeleton />}>
        <BlogGrid posts={posts} />
      </Suspense>
    </div>
  )
}
