//旋转链表
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
    if(head==NULL||k<=0)return head;
	
	ListNode* fast=head,*slow=head;ListNode* pre=head;
    ListNode* tmp=head;int len=0;
    while(tmp)
    {
        len++;
        tmp=tmp->next;
    }
    k=k%len;
    if(k==0)return head;
	for(int i=1;i<k;i++)
	{
		fast=fast->next;
	}
	while(fast->next)
	{
		fast=fast->next;
		pre=slow;
		slow=slow->next;
	}
	fast->next=head;pre->next=NULL;
	return slow;
        
    }
};
