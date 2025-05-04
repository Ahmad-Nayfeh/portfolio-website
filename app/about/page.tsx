import type { Metadata } from "next"
import Image from "next/image"
import Link from "next/link"
import { FileDown } from "lucide-react"
import { getAboutContent } from "@/lib/content"
import { Button } from "@/components/ui/button"
import Skills from "@/components/Skills"
import Experience from "@/components/Experience"
import Education from "@/components/Education"
import MarkdownRenderer from "@/components/MarkdownRenderer"

export const metadata: Metadata = {
  title: "About | Your Portfolio",
  description: "Learn more about me, my skills, and my experience",
}

export default async function AboutPage() {
  const aboutContent = await getAboutContent()

  return (
    <div className="container mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-8">About Me</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 mb-16">
        <div className="lg:col-span-1">
          <div className="relative aspect-square rounded-lg overflow-hidden mb-6">
            <Image src="/images/profile.jpg" alt="Profile Photo" fill className="object-cover" priority />
          </div>

          <Button asChild className="w-full">
            <a href="/resume.pdf" target="_blank">
              <FileDown size={16} className="mr-2" />
              Download Resume
            </a>
          </Button>
        </div>

        <div className="lg:col-span-2">
          <div className="prose dark:prose-invert max-w-none">
            <h2>Hello, I'm {aboutContent.name}</h2>
            <p className="text-xl mb-6">{aboutContent.tagline}</p>
            <MarkdownRenderer content={aboutContent.bio} />
          </div>
        </div>
      </div>

      <Skills skills={aboutContent.skills} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 mt-16">
        <Experience experience={aboutContent.experience} />
        <Education education={aboutContent.education} />
      </div>
    </div>
  )
}
