# Contribution Guide

## Feature Requests  
The checkmaite team encourages other JATIC teams to submit `checkmaite` feature requests by submitting an issue on the reference-implementation issue board using the [feature request template](https://gitlab.jatic.net/jatic/reference-implementation/reference-implementation/issues/new?issuable_template=feature-request). 

The checkmaite team periodically reviews these requests and then divides them according to scope and applicability across `checkmaite`:

- Features which enhance only a specific use-case, or require primarily niche familiarity of a JATIC component, are marked to be ***led by the appropriate JATIC team*** (e.g. the requestor's team). The checkmaite team can coordinate and support these cases. This marking is accomplished by adding the appropriate JATIC team's tag to the issue.  
- Features determined to be generally applicable to `checkmaite` are prioritized for development ***led by the checkmaite team itself.*** Other JATIC teams may volunteer to support or even lead these efforts. These issues are tracked within the checkmaite team's usual task workflow.

## Code-Compatibility Refactors 
The environment for `checkmaite` includes all of the JATIC tools as dependencies. Changes in `checkmaite` code may be required to support changes in these dependencies. The checkmaite team is ultimately responsible for ensuring `checkmaite` project correctly builds and operates with the set of dependency versions dictated by technical and security requirements.  

The checkmaite team requires the collaboration of JATIC teams which release components on which `checkmaite` depends:

- ***A bug or security vulnerability in a JATIC component.*** In this case, the checkmaite team will communicate with the JATIC Team responsible for the affected component in order to coordinate a solution and timeline for the fix.
- ***A new version of a JATIC component introduces breaking changes.*** The checkmaite team will communicate with the JATIC Team responsible for the component to understand the scope of the fix required. The checkmaite team can support minor compatibility changes, but will request assistance if substantial changes are required to support the new version. ***The checkmaite team requests maximum advanced notice when breaking API changes will be introduced in upcoming versions of JATIC components*** in order to reduce their impact. 


## External Contribution Procedure  
The checkmaite team requests that Contributing Teams follow this procedure:  
1.) Contributing Team commits to the `checkmaite` codebase within a feature branch of the checkmaite repo or their own fork of the repo  
2.) Contributing Team opens an MR to the checkmaite repo and adds the ***"needs:: contributing team review"*** tag  
3.) A second member of the Contributing Team completes a review of the MR  
4.) A member of the Contributing Team removes the "needs:: contributing team review" tag  
5.) A member of the Contributing Team adds the ***"needs: checkmaite team review"*** tag  

When the "needs: checkmaite team review" tag is added to an MR, the checkmaite team will schedule the MR review and communicate with the owner of the MR on any further changes.

# Setup Guide
For details on how to get setup as a developer, check out the [dev setup suide](dev_setup.md).
