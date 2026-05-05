import type { Metadata } from "next"
import BlogGrid from "@/components/BlogGrid"
import { getAllPosts } from "@/lib/content"

export const metadata: Metadata = {
  title: "The Notebook | Ahmad Nayfeh",
  description:
    "Notes on engineering, AI papers, and what I learn from building real systems.",
}

export default async function BlogPage() {
  const posts = await getAllPosts()

  return (
    <div className="mx-auto w-full max-w-[1400px] px-6 pb-24 pt-12 md:px-10 lg:px-16">
      {/* Editorial masthead for the list page. */}
      <header className="mb-14 border-b border-border pb-8">
        <div className="mb-5 flex items-center gap-3 animate-fade-up">
          <span aria-hidden className="h-px w-8 bg-accent" />
          <span className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            The Notebook · {posts.length} entr{posts.length === 1 ? "y" : "ies"}
          </span>
        </div>
        <div className="grid grid-cols-12 items-end gap-x-6 gap-y-4">
          <h1
            className="col-span-12 font-display text-display-xl text-balance lg:col-span-8 animate-fade-up [animation-delay:60ms]"
            style={{ animationFillMode: "both" }}
          >
            Notes from the workbench
          </h1>
          <p
            className="col-span-12 max-w-md text-base text-muted-foreground lg:col-span-4 animate-fade-up [animation-delay:120ms]"
            style={{ animationFillMode: "both" }}
          >
            A growing archive of writing on AI papers, engineering practice,
            and the small surprises that show up when you actually build
            things.
          </p>
        </div>
      </header>

      <BlogGrid posts={posts} />
    </div>
  )
}
