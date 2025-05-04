import type { Metadata } from "next"
import ProjectGrid from "@/components/ProjectGrid"
import { getAllProjects } from "@/lib/content"

export const metadata: Metadata = {
  title: "Projects | Your Portfolio",
  description: "Showcase of my latest projects and work",
}

export default async function ProjectsPage() {
  const projects = await getAllProjects()

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">Projects</h1>
      <p className="text-lg mb-8 text-muted-foreground">A collection of my work, side projects, and experiments.</p>
      <ProjectGrid projects={projects} />
    </div>
  )
}
