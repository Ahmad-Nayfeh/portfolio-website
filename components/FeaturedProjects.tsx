import Link from "next/link"
import { ArrowRight } from "lucide-react"
import ProjectCard from "@/components/ProjectCard"
import type { Project } from "@/types"

interface FeaturedProjectsProps {
  projects: Project[]
}

export default function FeaturedProjects({ projects }: FeaturedProjectsProps) {
  return (
    <section className="py-16">
      <div className="flex justify-between items-center mb-8">
        <h2 className="text-3xl font-bold">Featured Projects</h2>
        <Link
          href="/projects"
          className="flex items-center text-muted-foreground hover:text-foreground transition-colors"
        >
          View all
          <ArrowRight className="ml-1 h-4 w-4" />
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.map((project) => (
          <ProjectCard key={project.slug} project={project} />
        ))}
      </div>
    </section>
  )
}
