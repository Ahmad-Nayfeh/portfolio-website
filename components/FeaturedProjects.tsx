import ProjectCard from "@/components/ProjectCard"
import SectionMasthead from "@/components/editorial/SectionMasthead"
import type { Project } from "@/types"

interface FeaturedProjectsProps {
  projects: Project[]
}

export default function FeaturedProjects({ projects }: FeaturedProjectsProps) {
  return (
    <section className="py-16 md:py-24">
      <SectionMasthead
        kicker="Selected Work"
        title="Things I have built and broken"
        description="A short list of systems, models, and side experiments — picked because each taught me something I still use."
        link={{ href: "/projects", label: "View all projects" }}
      />

      <div className="grid grid-cols-1 gap-x-6 gap-y-10 md:grid-cols-2 lg:grid-cols-3">
        {projects.map((project, idx) => (
          <ProjectCard key={project.slug} project={project} index={idx} />
        ))}
      </div>
    </section>
  )
}
