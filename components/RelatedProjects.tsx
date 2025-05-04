import ProjectCard from "@/components/ProjectCard"
import type { Project } from "@/types"

interface RelatedProjectsProps {
  projects: Project[]
}

export default function RelatedProjects({ projects }: RelatedProjectsProps) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {projects.map((project) => (
        <ProjectCard key={project.slug} project={project} />
      ))}
    </div>
  )
}
