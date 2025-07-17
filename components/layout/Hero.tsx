import Link from "next/link"
import { ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { HomeContent } from "@/types"
import MarkdownRenderer from "@/components/MarkdownRenderer"


interface HeroProps {
  content: HomeContent
}

export default function Hero({ content }: HeroProps) {
  return (
    <section className="py-20 md:py-28">
      <div className="max-w-3xl">
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6 animate-in fade-in">{content.title}</h1>
        <p className="text-xl md:text-2xl text-muted-foreground mb-8 animate-in fade-in">{content.subtitle}</p>
        <div className="prose dark:prose-invert max-w-none prose-lg animate-in fade-in">
          <MarkdownRenderer content={content.description} />
        </div>
        <div className="flex flex-col sm:flex-row gap-4 mt-8 animate-in fade-in">
          <Button asChild size="lg">
            <Link href="/projects">
              View My Work
              <ArrowRight className="ml-2 h-4 w-4" />
            </Link>
          </Button>
          <Button asChild variant="outline" size="lg">
            <Link href="/about">About Me</Link>
          </Button>
        </div>
      </div>
    </section>
  )
}