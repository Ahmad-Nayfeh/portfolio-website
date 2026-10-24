import type { Route } from "next"
import Link from "next/link"
import Image from "next/image"
import { ArrowUpRight } from "lucide-react"
import type { Project } from "@/types"
import { projectYear, splitTitle } from "@/components/projects/projectName"
import { cn } from "@/lib/utils"
import { projectTiers, tierOf } from "@/config/site"

interface ProjectCardProps {
  project: Project
  /** No longer shown; kept so older callers compile. */
  sheetNo?: string
  /** "wide" lays the image beside the text (used when a card sits alone). */
  layout?: "stacked" | "wide"
}

/** Drawing-sheet card used on the projects index, the home page, and related lists. */
export default function ProjectCard({ project, sheetNo, layout = "stacked" }: ProjectCardProps) {
  const { name, tagline } = splitTitle(project.title)
  const isArabic = project.lang === "ar"

  return (
    <Link
      href={`/projects/${project.slug}` as Route}
      className={cn(
        "sheet sheet-hover group h-full",
        layout === "wide" ? "grid grid-cols-1 md:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]" : "flex flex-col",
      )}
      dir={isArabic ? "rtl" : undefined}
      lang={isArabic ? "ar" : undefined}
    >
      <div className={cn("flex h-9 items-center justify-between border-b border-border px-4", layout === "wide" && "md:col-span-2")} dir="ltr">
        <span className="label text-[9.5px]">{projectTiers.find((t) => t.id === tierOf(project.category))?.name}</span>
        <span className="flex items-center gap-2">
          {project.status && (
            <>
              <span className="lamp scale-75" aria-hidden />
              <span className="label text-[9.5px]">{project.status}</span>
              <span aria-hidden className="text-border">|</span>
            </>
          )}
          <span className="label text-[9.5px]">{projectYear(project)}</span>
        </span>
      </div>

      <div
        className={cn(
          "relative aspect-[16/10] overflow-hidden border-b border-border bg-secondary",
          layout === "wide" && "md:aspect-auto md:min-h-[240px] md:border-b-0 md:border-e",
        )}
      >
        <Image
          src={project.coverImage}
          alt=""
          fill
          className="object-cover grayscale-[35%] transition duration-500 group-hover:scale-[1.02] group-hover:grayscale-0"
          sizes="(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 400px"
        />
      </div>

      <div className="flex flex-1 flex-col p-5">
        <h3 className="text-[1.2rem] font-semibold leading-snug tracking-tight">
          <span className="redline">{name}</span>
        </h3>
        {tagline && <p className="mt-1 text-[14.5px] font-medium text-foreground/75">{tagline}</p>}
        <p className="mt-3 line-clamp-3 text-[14.5px] leading-relaxed text-muted-foreground">{project.excerpt}</p>
        <div className="mt-auto flex items-end justify-between gap-3 pt-5" dir="ltr">
          <div className="flex flex-wrap gap-1.5">
            {project.tags.slice(0, 3).map((t) => (
              <span key={t} className="chip">
                {t}
              </span>
            ))}
          </div>
          <ArrowUpRight size={17} className="shrink-0 text-muted-foreground transition-colors group-hover:text-accent" />
        </div>
      </div>
    </Link>
  )
}
