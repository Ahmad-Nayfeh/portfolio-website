import ProjectCard from "@/components/ProjectCard"
import type { Project } from "@/types"

export default function RelatedProjects({ projects }: { projects: Project[] }) {
  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
      {projects.map((project) => (
        <ProjectCard key={project.slug} project={project} />
      ))}
    </div>
  )
}
