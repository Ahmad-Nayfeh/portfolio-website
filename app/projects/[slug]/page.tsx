import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import { notFound } from "next/navigation"
import { ArrowLeft, Github, ExternalLink } from "lucide-react"
import { getProjectBySlug, getRelatedProjects, getAllProjects } from "@/lib/content"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { formatDate } from "@/lib/utils"
import MarkdownRenderer from "@/components/MarkdownRenderer"
import type { FullProject } from "@/types"
import RelatedProjects from "@/components/RelatedProjects"

// Define the parameter structure for the page
interface PageProps {
  params: Promise<{
    slug: string;
  }>;
}

// Generate static paths for all projects to improve performance
export async function generateStaticParams() {
  const projects = await getAllProjects();
  return projects.map((project) => ({
    slug: project.slug,
  }));
}

// Generate metadata for the page (for SEO)
export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const project = await getProjectBySlug(slug);

  if (!project) {
    return {
      title: "Project Not Found",
    };
  }

  return {
    title: `${project.title} | Projects`,
    description: project.excerpt,
  };
}

// The main page component
export default async function ProjectPage({ params }: PageProps) {
  const { slug } = await params;
  const project: FullProject | null = await getProjectBySlug(slug);

  if (!project) {
    notFound();
  }

  const relatedProjects = await getRelatedProjects(project, 3);

  // Check the language property of the project and set the direction
  const direction = project.lang === "ar" ? "rtl" : "ltr";

  return (
    // Apply the language direction to the main container
    <div className="container mx-auto px-4 py-12" dir={direction}>
      <Link
        href="/projects"
        className="mb-8 inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
      >
        <ArrowLeft size={16} />
        <span>Back to Projects</span>
      </Link>

      <article className="max-w-3xl mx-auto">
        <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl mb-4">
          {project.title}
        </h1>

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-8 text-sm text-muted-foreground">
          {project.date && <time dateTime={project.date}>{formatDate(project.date)}</time>}
          <div className="flex items-center gap-2">
             {project.githubLink && (
               <Button asChild variant="outline" size="sm">
                 <a href={project.githubLink} target="_blank" rel="noopener noreferrer">
                   <span className="flex items-center">
                     <Github size={16} className="mr-2" />
                     <span>Source Code</span>
                   </span>
                 </a>
               </Button>
             )}
             {project.liveDemoUrl && project.liveDemoUrl !== "#" && (
                <Button asChild variant="outline" size="sm">
                  <a href={project.liveDemoUrl} target="_blank" rel="noopener noreferrer">
                    <span className="flex items-center">
                      <ExternalLink size={16} className="mr-2" />
                      <span>Live Demo</span>
                    </span>
                  </a>
                </Button>
              )}
          </div>
        </div>

        {project.coverImage && (
          <div className="relative aspect-video mb-8 rounded-lg overflow-hidden border">
            <Image
              src={project.coverImage}
              alt={project.title}
              fill
              className="object-cover"
              priority
              sizes="(max-width: 768px) 100vw, (max-width: 1024px) 75vw, 800px"
            />
          </div>
        )}

        <div className="flex flex-wrap gap-2 mb-8">
          {project.tags?.map((tag) => (
            <Badge key={tag} variant="secondary">
              {tag}
            </Badge>
          ))}
        </div>

        <div className="grid gap-8 mb-8">
          {project.challenge && (
            <div>
              <h2 className="text-2xl font-bold mb-2">The Challenge</h2>
              <p>{project.challenge}</p>
            </div>
          )}
          {project.solution && (
            <div>
              <h2 className="text-2xl font-bold mb-2">The Solution</h2>
              <p>{project.solution}</p>
            </div>
          )}
          {project.technologies && project.technologies.length > 0 && (
            <div>
              <h2 className="text-2xl font-bold mb-2">Technologies Used</h2>
              <div className="flex flex-wrap gap-2">
                {project.technologies.map((tech) => (
                  <Badge key={tech}>{tech}</Badge>
                ))}
              </div>
            </div>
          )}
          {project.features && project.features.length > 0 && (
            <div>
              <h2 className="text-2xl font-bold mb-2">Key Features</h2>
              <ul className="list-disc pl-5 space-y-1">
                {project.features.map((feature, index) => (
                  <li key={index}>{feature}</li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {project.content && <MarkdownRenderer content={project.content} />}
      </article>

      {relatedProjects.length > 0 && (
        <section className="max-w-3xl mx-auto mt-16 pt-8 border-t">
          <h2 className="text-2xl font-bold mb-8">Related Projects</h2>
          <RelatedProjects projects={relatedProjects} />
        </section>
      )}
    </div>
  )
}
