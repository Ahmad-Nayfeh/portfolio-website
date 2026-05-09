"use client"

import { useState } from "react"
import ProjectCard from "@/components/ProjectCard"
import StaggerGroup from "@/components/ui/StaggerGroup"
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
      {/* Filter chips */}
      {allTags.length > 0 && (
        <div className="mb-12">
          <div className="mb-3 font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            Stack
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setSelectedTag(null)}
              className={cn(
                "rounded-full border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.15em] transition-all duration-200",
                selectedTag === null
                  ? "border-transparent"
                  : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
              )}
              style={
                selectedTag === null
                  ? { backgroundColor: "hsl(var(--accent))", color: "hsl(var(--primary-foreground))" }
                  : {}
              }
            >
              All
            </button>
            {allTags.map((tag) => (
              <button
                type="button"
                key={tag}
                onClick={() => setSelectedTag(tag)}
                className={cn(
                  "rounded-full border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.15em] transition-all duration-200",
                  selectedTag === tag
                    ? "border-transparent"
                    : "border-border text-muted-foreground hover:border-foreground/30 hover:text-foreground",
                )}
                style={
                  selectedTag === tag
                    ? { backgroundColor: "hsl(var(--accent))", color: "hsl(var(--primary-foreground))" }
                    : {}
                }
              >
                {tag}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Card grid with staggered entrance */}
      {filtered.length > 0 ? (
        <StaggerGroup
          staggerDelay={100}
          direction="up"
          distance={16}
          className="grid grid-cols-1 gap-x-6 gap-y-12 md:grid-cols-2 lg:grid-cols-3"
        >
          {filtered.map((project) => (
            <ProjectCard key={project.slug} project={project} />
          ))}
        </StaggerGroup>
      ) : (
        <div className="border-y py-20 text-center" style={{ borderColor: "hsl(var(--border))" }}>
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-muted-foreground">
            No projects match the current filter.
          </p>
        </div>
      )}
    </div>
  )
}
