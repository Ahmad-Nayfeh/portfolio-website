"use client"

import { useState } from "react"
import ProjectCard from "@/components/ProjectCard"
import type { Project } from "@/types"
import { Badge } from "@/components/ui/badge"

interface ProjectGridProps {
  projects: Project[]
}

export default function ProjectGrid({ projects }: ProjectGridProps) {
  const [selectedTag, setSelectedTag] = useState<string | null>(null)

  // Extract all unique tags from projects
  const allTags = Array.from(new Set(projects.flatMap((project) => project.tags))).sort()

  // Filter projects by selected tag
  const filteredProjects = selectedTag ? projects.filter((project) => project.tags.includes(selectedTag)) : projects

  return (
    <div>
      <div className="mb-8">
        <div className="flex flex-wrap gap-2">
          <Badge
            variant={selectedTag === null ? "default" : "outline"}
            className="cursor-pointer"
            onClick={() => setSelectedTag(null)}
          >
            All
          </Badge>
          {allTags.map((tag) => (
            <Badge
              key={tag}
              variant={selectedTag === tag ? "default" : "outline"}
              className="cursor-pointer"
              onClick={() => setSelectedTag(tag)}
            >
              {tag}
            </Badge>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProjects.map((project) => (
          <ProjectCard key={project.slug} project={project} />
        ))}
      </div>

      {filteredProjects.length === 0 && (
        <div className="text-center py-12">
          <p className="text-muted-foreground">No projects found with the selected tag.</p>
        </div>
      )}
    </div>
  )
}
