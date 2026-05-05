"use client"

import { useState } from "react"
import ProjectCard from "@/components/ProjectCard"
import type { Project } from "@/types"
import { cn } from "@/lib/utils"

interface ProjectGridProps {
  projects: Project[]
}

export default function ProjectGrid({ projects }: ProjectGridProps) {
  const [selectedTag, setSelectedTag] = useState<string | null>(null)

  const allTags = Array.from(new Set(projects.flatMap((p) => p.tags))).sort()
  const filtered = selectedTag
    ? projects.filter((p) => p.tags.includes(selectedTag))
    : projects

  return (
    <div>
      {allTags.length > 0 && (
        <div className="mb-12 border-b border-border pb-8">
          <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Filter by stack
          </div>
          <div className="flex flex-wrap gap-x-3 gap-y-1.5 font-mono text-[11px] uppercase tracking-[0.18em]">
            <button
              type="button"
              className={cn(
                "transition-colors",
                selectedTag === null
                  ? "text-accent"
                  : "text-muted-foreground hover:text-foreground",
              )}
              onClick={() => setSelectedTag(null)}
            >
              All
            </button>
            {allTags.map((tag) => (
              <button
                type="button"
                key={tag}
                className={cn(
                  "transition-colors",
                  selectedTag === tag
                    ? "text-accent"
                    : "text-muted-foreground hover:text-foreground",
                )}
                onClick={() => setSelectedTag(tag)}
              >
                #{tag}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3">
        {filtered.map((project, idx) => (
          <ProjectCard key={project.slug} project={project} index={idx} />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="border-y border-border py-20 text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            No projects match the current filter.
          </p>
        </div>
      )}
    </div>
  )
}
