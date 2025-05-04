import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { EducationItem } from "@/types"

interface EducationProps {
  education: EducationItem[]
}

export default function Education({
  education = [
    {
      degree: "Master of Computer Science",
      institution: "University of Technology",
      location: "San Francisco, CA",
      startDate: "2016",
      endDate: "2018",
      description: "Specialized in Human-Computer Interaction and Web Technologies.",
    },
    {
      degree: "Bachelor of Science in Computer Science",
      institution: "State University",
      location: "Boston, MA",
      startDate: "2012",
      endDate: "2016",
      description: "Graduated with honors. Focused on software engineering and web development.",
    },
  ],
}: EducationProps) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Education</h2>
      <div className="space-y-4">
        {education.map((item, index) => (
          <Card key={index}>
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">{item.degree}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-2">
                <div className="font-medium">{item.institution}</div>
                <div className="text-sm text-muted-foreground">{item.location}</div>
                <div className="text-sm text-muted-foreground">
                  {item.startDate} — {item.endDate}
                </div>
              </div>
              <p className="text-sm">{item.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
