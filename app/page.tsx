import Hero from "@/components/layout/Hero"
import FeaturedProjects from "@/components/FeaturedProjects"
import RecentPosts from "@/components/RecentPosts"
import Skills from "@/components/Skills"
import CTA from "@/components/CTA"
import { getHomeContent, getFeaturedProjects, getRecentPosts } from "@/lib/content"

export default async function Home() {
  const homeContent = await getHomeContent()
  const featuredProjects = await getFeaturedProjects(6)
  const recentPosts = await getRecentPosts(3)

  return (
    <div className="mx-auto w-full max-w-[1400px] px-6 md:px-10 lg:px-16">
      <Hero content={homeContent} />

      {/* The page rhythm intentionally alternates white-on-white with the
          quiet ruled sections below — keeps the eye moving the way a
          long-form magazine spread does. */}
      <FeaturedProjects projects={featuredProjects} />
      <RecentPosts posts={recentPosts} />
      <Skills />
      <CTA />
    </div>
  )
}
