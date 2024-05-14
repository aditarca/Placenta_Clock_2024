#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool
label: Check that submission is a Docker image

requirements:
- class: InlineJavascriptRequirement
- class: InitialWorkDirRequirement
  listing:
  - entryname: check_submission_type.py
    entry: |
      #!/usr/bin/env python
      import argparse
      import json

      parser = argparse.ArgumentParser()
      parser.add_argument("-e", "--entity_type", required=True, help="Entity type")
      args = parser.parse_args()

      if args.entity_type == "org.sagebionetworks.repo.model.docker.DockerRepository":
        status = "VALIDATED"
        errors = ""
      else:
        status = "INVALID"
        errors = "Submission must be a Docker image."

      with open("results.json", "w") as o:
        o.write(json.dumps({
            "submission_status": status,
            "submission_errors": errors
        }))

inputs:
- id: entity_type
  type: string
  inputBinding:
    prefix: -e

outputs:
- id: results
  type: File
  outputBinding:
    glob: results.json
- id: status
  type: string
  outputBinding:
    glob: results.json
    outputEval: $(JSON.parse(self[0].contents)['submission_status'])
    loadContents: true
- id: invalid_reasons
  type: string
  outputBinding:
    glob: results.json
    outputEval: $(JSON.parse(self[0].contents)['submission_errors'])
    loadContents: true

baseCommand:
- python3
- check_submission_type.py
