import Hero from "@/components/layout/Hero"
import FeaturedProjects from "@/components/FeaturedProjects"
import RecentPosts from "@/components/RecentPosts"
import Skills from "@/components/Skills"
import CTA from "@/components/CTA"
import { getHomeContent, getFeaturedProjects, getRecentPosts } from "@/lib/content"

export default async function Home() {
  const homeContent = await getHomeContent()
  const featuredProjects = await getFeaturedProjects(4)
  const recentPosts = await getRecentPosts(3)

  return (
    <div className="container mx-auto px-4 py-8">
      <Hero content={homeContent} />
      <FeaturedProjects projects={featuredProjects} />
      <Skills />
      <RecentPosts posts={recentPosts} />
      <CTA />
    </div>
  )
}
