import Image from "next/image"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import type { Project } from "@/types"
import { formatDate } from "@/lib/utils"

interface ProjectCardProps {
  project: Project
}

export default function ProjectCard({ project }: ProjectCardProps) {
  return (
    <Card className="overflow-hidden transition-all hover:shadow-md">
      <Link href={`/projects/${project.slug}`} className="block">
        <div className="relative aspect-video overflow-hidden">
          <Image
            src={project.coverImage || "/placeholder.svg"}
            alt={project.title}
            fill
            className="object-cover transition-transform hover:scale-105"
          />
        </div>
      </Link>

      <CardContent className="p-4">
        <div className="mb-2">
          <time dateTime={project.date} className="text-sm text-muted-foreground">
            {formatDate(project.date)}
          </time>
        </div>

        <Link href={`/projects/${project.slug}`} className="block">
          <h3 className="text-xl font-bold mb-2 hover:underline">{project.title}</h3>
        </Link>

        <p className="text-muted-foreground line-clamp-2 mb-4">{project.excerpt}</p>

        <div className="flex flex-wrap gap-2">
          {project.tags.slice(0, 3).map((tag) => (
            <Badge key={tag} variant="secondary" className="text-xs">
              {tag}
            </Badge>
          ))}
          {project.tags.length > 3 && (
            <Badge variant="outline" className="text-xs">
              +{project.tags.length - 3} more
            </Badge>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
