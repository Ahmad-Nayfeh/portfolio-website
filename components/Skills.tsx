import { Badge } from "@/components/ui/badge"

interface SkillsProps {
  skills?: string[]
}

export default function Skills({
  skills = [
    "Python",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision",
    "Signal Processing",
    "Image Processing",
    "Data Preprocessing",
    "Data Analysis",
    "Data Visualization",
  ],
}: SkillsProps) {
  return (
    <section className="py-16">
      <h2 className="text-3xl font-bold mb-8">Skills & Technologies</h2>

      <div className="flex flex-wrap gap-2">
        {skills.map((skill) => (
          <Badge key={skill} variant="secondary" className="text-sm py-1 px-3">
            {skill}
          </Badge>
        ))}
      </div>
    </section>
  )
}
