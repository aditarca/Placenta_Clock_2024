#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool
label: Score Predictions

requirements:
- class: InlineJavascriptRequirement

inputs:
- id: input_file
  type: File
- id: goldstandard
  type: File
- id: task_number
  type: string
- id: check_validation_finished
  type: boolean?

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
- id: true_scores
  type: File
  outputBinding:
    glob: true_results.json

baseCommand: score.R
arguments:
- prefix: -p
  valueFrom: $(inputs.input_file.path)
- prefix: -g
  valueFrom: $(inputs.goldstandard.path)
- prefix: -t
  valueFrom: $(inputs.task_number)
- prefix: -o
  valueFrom: results.json

hints:
  DockerRequirement:
    dockerPull: docker.synapse.org/syn59425204/
